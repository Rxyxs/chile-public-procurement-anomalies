"""
Anomalies in Chilean public procurement, on every purchase order of Mercado Público.

Three analyses on 32 months of real ChileCompra purchase orders (January-November 2024 and
January 2025-September 2026):

1. Bunching: how many more Compra Ágil orders sit just under the legal cap than a smooth
   trend predicts, before and after Ley 21.634 moved the cap from 30 to 100 UTM.
2. Split purchases: whether orders at the cap come with another order of the same buyer
   and supplier within a week more often than other orders of the same buyer.
3. Detectors: a z-score rule, Isolation Forest and an autoencoder (three activations),
   trained on 2025 order lines and scored on anomalies planted in real 2026 lines.

    python main.py --download   # once: 32 monthly ZIPs (~3 GB) and the UTM into data/raw/
    python main.py              # ingest, analyse, write results/results.json and the figures
"""

from __future__ import annotations

import argparse
import json
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import polars as pl

from src import detection as det
from src.bunching import bunching, sensitivity
from src.ingest import PROCESSED_DIR, ingest_all
from src.orders import REGIMES, build_orders, compra_agil, load_utm, scan_lines
from src.plots import plot_bunching, plot_detector_recall, plot_placebos, plot_sibling_rates, plot_training_curves
from src.sources import MONTHS, download_all
from src.splitting import add_sibling_flags, near_cap_effect, sibling_rate_by_size, split_clusters

RESULTS = PROCESSED_DIR.parent.parent / "results" / "results.json"
PLACEBO_CAPS = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0)
TRAIN_SAMPLE = 400_000
TEST_SAMPLE = 200_000
IF_SAMPLE = 200_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="download the raw files that are missing")
    args = parser.parse_args()
    if args.download:
        download_all()
    ingest_all()

    lines = scan_lines()
    orders = build_orders(lines, load_utm(sorted({y for y, _ in MONTHS})))
    data = {
        "order_lines": int(lines.select(pl.len()).collect().item()),
        "orders": orders.height,
        "buyer_units": int(orders["buyer_unit"].n_unique()),
        "suppliers": int(orders["supplier"].n_unique()),
        "orders_over_one_trillion_clp": int(orders.filter(pl.col("total_clp") > 1e12).height),
    }
    print(data)

    # ---- 1. bunching -------------------------------------------------------------------
    amounts, fits, compra = {}, {}, {}
    for regime, (_, _, cap) in REGIMES.items():
        ag = compra_agil(orders, regime)
        compra[regime] = {"orders": ag.height, "over_cap": int(ag.filter(pl.col("total_utm") > cap * 1.001).height)}
        amounts[regime] = ag["total_utm"].to_numpy()
        fits[regime] = bunching(amounts[regime], cap)
        compra[regime]["sensitivity"] = sensitivity(amounts[regime], cap)
        print(regime, compra[regime]["orders"], round(fits[regime]["excess_mass"], 3))
    old_cap_after = bunching(amounts["cap100"], 30.0)
    placebos = {c: bunching(amounts["cap100"], c) for c in PLACEBO_CAPS}

    # ---- 2. split purchases ------------------------------------------------------------
    splitting, rates = {}, {}
    for regime, (_, _, cap) in REGIMES.items():
        ag = add_sibling_flags(compra_agil(orders, regime).filter(pl.col("total_utm") <= cap))
        by_size = sibling_rate_by_size(ag, cap)
        rates[regime] = (by_size["size_bin"].to_numpy(), by_size["sibling_rate"].to_numpy())
        splitting[regime] = {
            "sibling_rate": float(ag["has_sibling"].mean()),
            "within_buyer": near_cap_effect(ag, cap),
            "within_buyer_same_day": near_cap_effect(ag, cap, outcome="same_day_sibling"),
            "chains": split_clusters(ag, cap),
            "by_size": by_size.to_dicts(),
        }

    # ---- 3. detectors ------------------------------------------------------------------
    all_lines = det.add_history(det.clean_lines(lines))
    train = all_lines.filter(pl.col("created_date").is_between(*det.TRAIN))
    test = all_lines.filter(pl.col("created_date").is_between(*det.TEST))
    stats, buyers = det.group_stats(train), det.buyer_stats(train)
    f_train = det.build_features(train, stats, buyers)
    f_test = det.build_features(test, stats, buyers)
    coverage = {"train": f_train.height / train.height, "test": f_test.height / test.height}
    f_train = f_train.sample(TRAIN_SAMPLE, seed=det.SEED)
    sample = test.join(f_test.select("order_code", "line_id"), on=["order_code", "line_id"]).sample(TEST_SAMPLE, seed=det.SEED)
    planted = det.build_features(det.plant_anomalies(sample, stats), stats, buyers)
    X_train, X_test = det.standardize(f_train.select(det.FEATURES).to_numpy(), planted.select(det.FEATURES).to_numpy())
    labels = planted["anomaly"].to_list()

    recalls = {"Rule (z-scores)": det.recall_at_budget(det.rule_scores(planted), labels)}
    recalls["Isolation Forest"] = det.recall_at_budget(det.isolation_forest_scores(X_train[:IF_SAMPLE], X_test), labels)
    cut = int(len(X_train) * 0.85)
    curves = {}
    for name, activation in det.ACTIVATIONS.items():
        result = det.train_autoencoder(X_train[:cut], X_train[cut:], activation)
        recalls[f"Autoencoder ({name})"] = det.recall_at_budget(det.reconstruction_error(result.model, X_test), labels)
        curves[name] = (result.train_losses, result.val_losses, result.best_epoch)
        print(name, result.best_epoch, recalls[f"Autoencoder ({name})"]["overall"])
    for name, r in recalls.items():
        print(f"{name:<28} {r['overall']:.3f}")

    # ---- outputs -----------------------------------------------------------------------
    plot_bunching(amounts, {r: c for r, (_, _, c) in REGIMES.items()}, fits)
    plot_placebos({r: fits[r]["excess_mass"] for r in fits}, {c: v["excess_mass"] for c, v in placebos.items()})
    plot_sibling_rates(rates)
    plot_detector_recall(recalls, det.ANOMALY_TYPES)
    plot_training_curves(curves)

    def summary(fit: dict) -> dict:
        return {k: v for k, v in fit.items() if k not in ("centers", "counts", "predicted")}

    results = {
        "data": data,
        "compra_agil": {r: {**compra[r], "bunching": summary(fits[r])} for r in fits},
        "old_cap_30_after_change": summary(old_cap_after),
        "placebo_caps": {str(int(c)): summary(v) for c, v in placebos.items()},
        "splitting": splitting,
        "detectors": {
            "coverage": coverage,
            "train_lines": TRAIN_SAMPLE,
            "test_lines": TEST_SAMPLE,
            "planted": int(sum(x is not None for x in labels)),
            "recall_at_5pct": recalls,
            "autoencoder_best_epoch": {n: c[2] for n, c in curves.items()},
        },
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"written {RESULTS}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
