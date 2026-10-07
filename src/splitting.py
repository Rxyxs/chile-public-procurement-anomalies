"""Looking for split purchases directly: Compra Ágil orders that come with a sibling.

A *sibling* is another Compra Ágil order from the same buying unit to the same supplier,
created within ``WINDOW_DAYS`` days. Splitting one purchase in two to stay under the cap
creates exactly that pattern, but so do ordinary repeat purchases (food, office supplies),
so the question is whether siblings are *more frequent right under the cap* than for
orders of other sizes, comparing orders of the same buying unit (buyer fixed effects).

None of this proves an order was split: it ranks where an auditor should look first.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm

WINDOW_DAYS = 7
NEAR_CAP = 0.95  # orders at 95-100% of the cap ...
REFERENCE = (0.60, 0.90)  # ... compared with orders at 60-90% of it
PAIR = ["buyer_unit", "supplier"]


def add_sibling_flags(orders: pl.DataFrame, window_days: int = WINDOW_DAYS) -> pl.DataFrame:
    """``has_sibling``: another order of the same buyer and supplier within ``window_days``;
    ``same_day_sibling``: one created the same day; ``gap_to_previous``: days since the
    pair's previous order (null for its first)."""
    o = orders.sort(PAIR + ["created_date", "order_code"])
    prev_gap = (pl.col("created_date") - pl.col("created_date").shift(1)).over(PAIR).dt.total_days()
    next_gap = (pl.col("created_date").shift(-1) - pl.col("created_date")).over(PAIR).dt.total_days()
    o = o.with_columns(prev_gap.alias("gap_to_previous"), next_gap.alias("_next_gap"))
    nearest = pl.min_horizontal("gap_to_previous", "_next_gap")
    return o.with_columns(
        (nearest <= window_days).fill_null(False).alias("has_sibling"),
        (nearest == 0).fill_null(False).alias("same_day_sibling"),
    ).drop("_next_gap")


def sibling_rate_by_size(orders: pl.DataFrame, cap: float, width: float = 0.05) -> pl.DataFrame:
    """Share of orders with a sibling by order size, in bins of ``width`` times the cap."""
    rel = pl.col("total_utm") / cap
    return (
        orders.filter(rel <= 1.0)
        .with_columns(((rel / width).floor() * width).clip(0, 1 - width).round(4).alias("size_bin"))
        .group_by("size_bin")
        .agg(
            pl.len().alias("orders"),
            pl.col("has_sibling").mean().alias("sibling_rate"),
            pl.col("same_day_sibling").mean().alias("same_day_rate"),
        )
        .sort("size_bin")
    )


def near_cap_effect(
    orders: pl.DataFrame, cap: float, near: float = NEAR_CAP, reference: tuple[float, float] = REFERENCE,
    outcome: str = "has_sibling",
) -> dict:
    """Linear probability model of ``outcome`` on "the order sits at 95-100% of the cap",
    against orders at 60-90%, with buying-unit fixed effects and errors clustered by
    buying unit. The coefficient is in probability points."""
    rel = pl.col("total_utm") / cap
    sample = orders.filter(rel.is_between(reference[0], reference[1], closed="left") | rel.is_between(near, 1.0))
    sample = sample.with_columns((rel >= near).cast(pl.Float64).alias("near"), pl.col(outcome).cast(pl.Float64).alias("y"))
    # buyers with orders on both sides are the only ones that identify the within-buyer gap
    both = sample.group_by("buyer_unit").agg(pl.col("near").min().alias("lo"), pl.col("near").max().alias("hi"))
    sample = sample.join(both.filter(pl.col("lo") < pl.col("hi")).select("buyer_unit"), on="buyer_unit")
    demeaned = sample.with_columns(
        (pl.col("y") - pl.col("y").mean().over("buyer_unit")).alias("y_dm"),
        (pl.col("near") - pl.col("near").mean().over("buyer_unit")).alias("near_dm"),
    )
    groups = demeaned["buyer_unit"].cast(pl.Categorical).to_physical().to_numpy()
    fit = sm.OLS(demeaned["y_dm"].to_numpy(), demeaned["near_dm"].to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    low, high = fit.conf_int()[0]
    raw = sample.group_by("near").agg(pl.col("y").mean()).sort("near")["y"].to_list()
    return {
        "outcome": outcome,
        "effect": float(fit.params[0]),
        "ci95": [float(low), float(high)],
        "p_value": float(fit.pvalues[0]),
        "rate_reference": float(raw[0]),
        "rate_near_cap": float(raw[1]),
        "orders": int(sample.height),
        "buyers": int(sample["buyer_unit"].n_unique()),
    }


def split_clusters(orders: pl.DataFrame, cap: float, gap_days: int = WINDOW_DAYS) -> dict:
    """Chains of orders of the same buyer and supplier, each at most ``gap_days`` after the
    previous one, whose orders are all under the cap but add up to more than it: the
    purchases that would have needed another procedure had they been made as one."""
    if "gap_to_previous" not in orders.columns:
        orders = add_sibling_flags(orders, gap_days)
    o = orders.sort(PAIR + ["created_date", "order_code"]).with_columns(
        (pl.col("gap_to_previous").is_null() | (pl.col("gap_to_previous") > gap_days)).cum_sum().alias("chain")
    )
    chains = o.group_by("chain").agg(
        pl.len().alias("size"), pl.col("total_utm").sum().alias("sum_utm"), pl.col("total_utm").max().alias("max_utm")
    )
    flagged = chains.filter((pl.col("size") >= 2) & (pl.col("sum_utm") > cap) & (pl.col("max_utm") <= cap))
    total_utm = float(o["total_utm"].sum())
    return {
        "chains_over_cap": int(flagged.height),
        "orders_in_them": int(flagged["size"].sum()),
        "share_of_orders": float(flagged["size"].sum() / o.height),
        "utm_in_them": float(flagged["sum_utm"].sum()),
        "share_of_amount": float(flagged["sum_utm"].sum() / total_utm),
        "median_orders_per_chain": float(np.median(flagged["size"].to_numpy())) if flagged.height else 0.0,
    }
