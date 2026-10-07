"""One row per purchase order, with its amount in UTM and the Compra Ágil cap that applied.

The cap is the one in force when an order was *created*, not when it was sent: an order
created in November 2024 and sent in January 2025 was bought under the old rule. Ley 21.634
raised the Compra Ágil cap from 30 to 100 UTM on 12 December 2024; the cap binds on the
total including taxes (no Compra Ágil order goes above 100 UTM gross, while net amounts stop
near 84 UTM, 100 / 1.19). Amounts are converted with the UTM of the creation month.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl

from src.ingest import PROCESSED_DIR, processed_path
from src.sources import MONTHS, RAW_DIR, utm_path

CAP_CHANGE = dt.date(2024, 12, 12)
REGIMES = {
    # name: (first creation date, last creation date, Compra Ágil cap in UTM)
    "cap30": (dt.date(2024, 1, 1), dt.date(2024, 11, 30), 30.0),
    "cap100": (dt.date(2025, 1, 1), dt.date(2026, 9, 30), 100.0),
}
COMPRA_AGIL = "AG"


def load_utm(years: list[int], raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """``month`` (first day) and ``utm`` (pesos) from mindicador's yearly files."""
    rows = []
    for year in years:
        for item in json.loads(utm_path(year, raw_dir).read_text(encoding="utf-8"))["serie"]:
            day = dt.date.fromisoformat(item["fecha"][:10])
            rows.append((dt.date(day.year, day.month, 1), float(item["valor"])))
    utm = pl.DataFrame(rows, schema={"month": pl.Date, "utm": pl.Float64}, orient="row").unique("month").sort("month")
    return utm


def scan_lines(months: list[tuple[int, int]] = MONTHS, processed_dir: Path = PROCESSED_DIR) -> pl.LazyFrame:
    return pl.scan_parquet([processed_path(y, m, processed_dir) for y, m in months])


def build_orders(lines: pl.LazyFrame, utm: pl.DataFrame) -> pl.DataFrame:
    """Collapses order lines into orders. An order that appears in more than one monthly
    file (sent again after a change) keeps its latest version."""
    orders = (
        lines.sort("sent_date")
        .group_by("order_code")
        .agg(
            pl.col("order_type").last(),
            pl.col("status").last(),
            pl.col("created_date").last(),
            pl.col("sent_date").last(),
            pl.col("currency").last(),
            pl.col("total_clp").last(),
            pl.col("buyer_unit").last(),
            pl.col("buyer_org").last(),
            pl.col("sector").last(),
            pl.col("buyer_region").last(),
            pl.col("supplier").last(),
            pl.col("category_code").sort_by("line_total", descending=True).first().alias("main_category"),
            pl.len().alias("n_lines"),
        )
        .collect()
    )
    orders = orders.with_columns(pl.col("created_date").dt.truncate("1mo").alias("month")).join(utm, on="month", how="left")
    regime = pl.lit(None, dtype=pl.String)
    for name, (start, end, _) in REGIMES.items():
        regime = pl.when(pl.col("created_date").is_between(start, end)).then(pl.lit(name)).otherwise(regime)
    cap = pl.lit(None, dtype=pl.Float64)
    for name, (_, _, value) in REGIMES.items():
        cap = pl.when(pl.col("regime") == name).then(pl.lit(value)).otherwise(cap)
    return (
        orders.with_columns((pl.col("total_clp") / pl.col("utm")).alias("total_utm"), regime.alias("regime"))
        .with_columns(cap.alias("cap_utm"))
        .drop("month")
    )


def compra_agil(orders: pl.DataFrame, regime: str) -> pl.DataFrame:
    """Compra Ágil orders created in ``regime``, in pesos, with a positive amount."""
    return orders.filter(
        (pl.col("order_type") == COMPRA_AGIL)
        & (pl.col("regime") == regime)
        & (pl.col("currency") == "CLP")
        & (pl.col("total_utm") > 0)
    )
