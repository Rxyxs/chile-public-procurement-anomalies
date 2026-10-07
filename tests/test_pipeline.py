"""Offline tests for every stage of the pipeline, on small fixtures.

The results in the README come from the real ChileCompra files; these only check that
each step does what its docstring says.
"""

import datetime as dt
import io
import json
import zipfile

import numpy as np
import polars as pl
import pytest

from src import detection as det
from src.bunching import bin_counts, bunching
from src.ingest import COLUMNS, decode_mixed, ingest_month, parse_month
from src.orders import build_orders, compra_agil, load_utm
from src.sources import oc_path, utm_path
from src.splitting import add_sibling_flags, near_cap_effect, split_clusters

# ----------------------------------------------------------------------------- ingest


def _csv(rows: list[dict]) -> bytes:
    header = ";".join(f'"{c}"' for c in COLUMNS)
    body = []
    for r in rows:
        body.append(";".join(r.get(c, "NA") for c in COLUMNS))
    return (header + "\n" + "\n".join(body) + "\n").encode("cp1252", errors="strict")


def test_decode_mixed_reads_utf8_and_windows_1252_in_the_same_file():
    raw = "Aeródromo".encode("cp1252") + b" " + "ISOFÁNICA".encode("utf-8") + b" " + "“x”".encode("cp1252")
    assert decode_mixed(raw) == "Aeródromo ISOFÁNICA “x”"


def test_parse_month_handles_comma_decimals_multiline_text_and_na():
    row = {
        "Codigo": '"1-1-AG25"', "Tipo": '"AG"', "EsTratoDirecto": '"Si"', "Estado": '"Aceptada"',
        "FechaCreacion": '"2025-01-03"', "FechaEnvio": '"2025-01-05"', "TipoMonedaOC": '"CLP"',
        "MontoTotalOC_PesosChilenos": "119000", "TotalNetoOC": "100000", "CodigoUnidadCompra": "7",
        "CodigoProveedor": "99", "IDItem": "1", "codigoProductoONU": "14111509", "RubroN1": '"Línea\ndos"',
        "cantidad": "2,5", "UnidadMedida": '"Unidad"', "monedaItem": '"CLP"', "precioNeto": "40000",
        "totalLineaNeto": "100000",
    }
    df = parse_month(_csv([row]))
    assert df.height == 1
    assert df["quantity"][0] == 2.5
    assert df["created_date"][0] == dt.date(2025, 1, 3)
    assert df["is_direct"][0] is True
    assert df["buyer_region"][0] is None  # "NA"
    assert df["rubro"][0] == "Línea\ndos"


def test_ingest_month_reads_the_zip_once(tmp_path):
    row = {"Codigo": '"1-1-AG25"', "Tipo": '"AG"', "FechaCreacion": '"2025-01-03"', "FechaEnvio": '"2025-01-05"',
           "IDItem": "1", "cantidad": "1", "precioNeto": "10", "totalLineaNeto": "10"}
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "out"
    raw_dir.mkdir()
    with zipfile.ZipFile(oc_path(2025, 1, raw_dir), "w") as z:
        z.writestr("2025-1.csv", _csv([row]))
    path = ingest_month(2025, 1, raw_dir, out_dir)
    assert pl.read_parquet(path).height == 1
    assert ingest_month(2025, 1, raw_dir, out_dir) == path  # cached


# ----------------------------------------------------------------------------- orders


def _lines(rows):
    base = {"order_type": "AG", "status": "Aceptada", "currency": "CLP", "buyer_org": "o", "sector": "s",
            "buyer_region": "r", "category_code": "c", "line_total": 1.0, "line_id": "1"}
    return pl.DataFrame([{**base, **r} for r in rows]).lazy()


def test_build_orders_uses_the_creation_month_utm_the_latest_version_and_the_cap_in_force(tmp_path):
    utm_path(2024, tmp_path).write_text(json.dumps({"serie": [{"fecha": "2024-11-01T03:00:00.000Z", "valor": 65000}]}))
    utm_path(2025, tmp_path).write_text(json.dumps({"serie": [{"fecha": "2025-01-01T03:00:00.000Z", "valor": 67000}]}))
    utm = load_utm([2024, 2025], tmp_path)
    lines = _lines([
        # created under the old rule, sent after the change: cap 30
        {"order_code": "A", "created_date": dt.date(2024, 11, 20), "sent_date": dt.date(2025, 1, 2),
         "total_clp": 65000.0 * 29, "buyer_unit": "b", "supplier": "s1"},
        # sent twice: the later version wins
        {"order_code": "B", "created_date": dt.date(2025, 1, 5), "sent_date": dt.date(2025, 1, 6),
         "total_clp": 1.0, "buyer_unit": "b", "supplier": "s1"},
        {"order_code": "B", "created_date": dt.date(2025, 1, 5), "sent_date": dt.date(2025, 1, 9),
         "total_clp": 67000.0 * 99, "buyer_unit": "b", "supplier": "s1"},
        # December 2024 belongs to neither regime
        {"order_code": "C", "created_date": dt.date(2024, 12, 15), "sent_date": dt.date(2025, 1, 2),
         "total_clp": 100.0, "buyer_unit": "b", "supplier": "s1"},
    ])
    orders = build_orders(lines, utm).sort("order_code")
    a, b, c = orders.to_dicts()
    assert a["regime"] == "cap30" and a["cap_utm"] == 30.0 and a["total_utm"] == pytest.approx(29.0)
    assert b["regime"] == "cap100" and b["total_utm"] == pytest.approx(99.0)
    assert c["regime"] is None
    assert compra_agil(orders, "cap100").height == 1


# ----------------------------------------------------------------------------- bunching


def test_bunching_is_zero_on_a_flat_density_and_recovers_a_planted_pile():
    rng = np.random.default_rng(0)
    flat = rng.uniform(40, 100, 200_000)
    assert abs(bunching(flat, 100.0, n_boot=50)["excess_mass"]) < 0.05
    pile = np.concatenate([flat, rng.uniform(99, 100, 20_000)])
    # 200,000 orders spread over 60 UTM leave ~33,300 in the last 10 UTM: +20,000 is b = 0.6
    r = bunching(pile, 100.0, n_boot=50)
    assert 0.55 < r["excess_mass"] < 0.65
    assert r["ci95"][0] < r["excess_mass"] < r["ci95"][1]


def test_bin_counts_cover_up_to_and_including_the_cap():
    centers, counts = bin_counts(np.array([59.9, 60.0, 99.99, 100.0, 100.5]), 100.0)
    assert counts.sum() == 3 and counts[-1] == 2  # an order of exactly 100 UTM is allowed


# ----------------------------------------------------------------------------- splitting


def _orders(rows):
    return pl.DataFrame(rows).with_columns(pl.col("created_date").cast(pl.Date))


def test_sibling_flags_and_chains_over_the_cap():
    d = dt.date(2025, 3, 3)
    o = _orders([
        {"order_code": "1", "buyer_unit": "b", "supplier": "s", "created_date": d, "total_utm": 60.0},
        {"order_code": "2", "buyer_unit": "b", "supplier": "s", "created_date": d + dt.timedelta(days=3), "total_utm": 60.0},
        {"order_code": "3", "buyer_unit": "b", "supplier": "s", "created_date": d + dt.timedelta(days=30), "total_utm": 60.0},
        {"order_code": "4", "buyer_unit": "b", "supplier": "t", "created_date": d, "total_utm": 99.0},
    ])
    flagged = add_sibling_flags(o).sort("order_code")
    assert flagged["has_sibling"].to_list() == [True, True, False, False]
    chains = split_clusters(flagged, 100.0)
    assert chains["chains_over_cap"] == 1 and chains["orders_in_them"] == 2


def test_near_cap_effect_compares_orders_of_the_same_buyer():
    rng = np.random.default_rng(1)
    rows = []
    for b in range(300):
        base = rng.uniform(0.1, 0.6)  # buyers differ a lot in how often they repeat
        for i in range(40):
            near = i % 2 == 0
            amount = rng.uniform(95, 100) if near else rng.uniform(60, 90)
            rows.append({"buyer_unit": str(b), "total_utm": amount,
                         "has_sibling": bool(rng.random() < base + (0.10 if near else 0.0))})
    r = near_cap_effect(pl.DataFrame(rows), 100.0)
    assert 0.06 < r["effect"] < 0.14
    assert r["ci95"][0] < 0.10 < r["ci95"][1]


# ----------------------------------------------------------------------------- detection


def _lines_for_detection(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    products = rng.integers(0, 20, n)
    price = np.exp(rng.normal(products * 0.5 + 8, 0.3))
    qty = np.exp(rng.normal(1, 0.5, n)).round() + 1
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=int(x)) for x in rng.integers(0, 600, n)]
    return pl.DataFrame({
        "order_code": [f"o{i}" for i in range(n)], "line_id": ["1"] * n, "created_date": dates,
        "order_type": ["AG"] * n, "buyer_unit": [f"b{x}" for x in rng.integers(0, 30, n)],
        "supplier": [f"s{x}" for x in rng.integers(0, 80, n)], "product_code": [str(p) for p in products],
        "unit": ["Unidad"] * n, "quantity": qty, "unit_price": price, "line_total": price * qty,
    })


def test_history_only_looks_backwards():
    lines = det.add_history(_lines_for_detection())
    assert (lines["supplier_age_days"] >= 0).all()
    first = lines.sort("created_date").group_by(["buyer_unit", "supplier"]).first()
    assert (first["pair_history"] == 0).all()


def test_planted_anomalies_change_what_they_say_and_stand_out():
    lines = det.add_history(_lines_for_detection())
    train = lines.filter(pl.col("created_date") < dt.date(2026, 1, 1))
    test = lines.filter(pl.col("created_date") >= dt.date(2026, 1, 1))
    stats = det.group_stats(train, min_lines=10)
    planted = det.plant_anomalies(test, stats, fraction=0.2)
    kinds = planted["anomaly"].to_numpy()
    digit = kinds == "digit_error"
    assert np.allclose(planted["unit_price"].to_numpy()[digit], test["unit_price"].to_numpy()[digit] * 1000)
    assert (planted["supplier_age_days"].to_numpy()[kinds == "new_supplier"] == 0).all()
    untouched = np.array([k is None for k in kinds])
    assert np.array_equal(planted["unit_price"].to_numpy()[untouched], test["unit_price"].to_numpy()[untouched])
    features = det.build_features(planted, stats, det.buyer_stats(train))
    r = det.recall_at_budget(det.rule_scores(features), features["anomaly"].to_list(), budget=0.2)
    assert r["digit_error"] > 0.9  # a thousandfold price is impossible to miss on clean data
    assert r["overall"] > 0.2  # better than a random 20% review


def test_recall_at_budget_counts_the_top_scores():
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    labels = [None, "digit_error", None, "overpricing"]
    r = det.recall_at_budget(scores, labels, budget=0.5)
    assert r["overall"] == 0.5 and r["digit_error"] == 1.0 and r["overpricing"] == 0.0 and r["precision"] == 0.5


def test_autoencoder_trains_and_scores_outliers_higher():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    X[:, 1] = X[:, 0] * 2  # structure the network can learn
    result = det.train_autoencoder(X[:1500], X[1500:], max_epochs=30, patience=5)
    normal = det.reconstruction_error(result.model, X[1500:]).mean()
    outlier = X[1500:].copy()
    outlier[:, 1] = -outlier[:, 1]  # breaks the structure
    assert det.reconstruction_error(result.model, outlier).mean() > normal
    assert 1 <= result.best_epoch <= 30
