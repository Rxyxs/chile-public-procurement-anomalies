"""Unsupervised detectors on real order lines, scored on anomalies planted in real lines.

Real purchase orders come without labels -- nobody marks a line as fraudulent -- so the
detectors are compared the way the first version of this project did, but on a real
background: five kinds of anomaly are planted in 5% of a sample of real 2026 lines, every
detector ranks all the lines, and the score is the share of planted lines in its top 5%.
The real lines keep whatever real anomalies they already had; those count against the
detectors, as they would in an audit.

Features are relative to each line's product (ONU code) and unit of measure, with median
and MAD from the 2025 lines: real prices have heavy tails, and a mean and a standard
deviation estimated on them would be dragged by the outliers the detectors look for.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from sklearn.ensemble import IsolationForest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
TRAIN = (dt.date(2025, 1, 1), dt.date(2025, 12, 31))
TEST = (dt.date(2026, 1, 1), dt.date(2026, 9, 30))
MIN_GROUP_LINES = 30
MAD_FLOOR = 0.05  # in log units: groups where every line has the same quantity have MAD 0
Z_CLIP = 20.0
FEATURES = ["price_z", "quantity_z", "total_z", "price_vs_buyer", "supplier_age_log", "pair_history_log"]
MIN_BUYER_LINES = 3
ANOMALY_TYPES = ["overpricing", "inflated_quantity", "digit_error", "new_supplier", "wrong_product"]
BUDGET = 0.05  # an auditor reviews the top 5% of lines

# autoencoder training
BATCH_SIZE = 1024
MAX_EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 1e-3
ACTIVATIONS: dict[str, type[nn.Module]] = {"ReLU": nn.ReLU, "GELU": nn.GELU, "Swish (SiLU)": nn.SiLU}


# ----------------------------------------------------------------------------- data


def clean_lines(lines: pl.LazyFrame) -> pl.DataFrame:
    """Lines in pesos with a positive price and quantity and a product code, one row per
    order line (the latest version if an order was sent twice)."""
    return (
        lines.filter(
            (pl.col("line_currency") == "CLP")
            & (pl.col("unit_price") > 0)
            & (pl.col("quantity") > 0)
            & (pl.col("line_total") > 0)
            & pl.col("supplier").is_not_null()
            & pl.col("product_code").is_not_null()
            & (pl.col("product_code") != "0")
            & pl.col("unit").is_not_null()
        )
        .sort("sent_date")
        .unique(["order_code", "line_id"], keep="last")
        .select(
            "order_code", "line_id", "created_date", "order_type", "buyer_unit", "supplier",
            "product_code", "unit", "quantity", "unit_price", "line_total",
        )
        .collect()
    )


def add_history(lines: pl.DataFrame) -> pl.DataFrame:
    """``supplier_age_days``: days since the supplier's first order in the data;
    ``pair_history``: orders of the same buyer and supplier created before this one.
    Both only look backwards, so they are available when the order is placed."""
    first_seen = lines.group_by("supplier").agg(pl.col("created_date").min().alias("first_seen"))
    orders = (
        lines.select("order_code", "created_date", "buyer_unit", "supplier")
        .unique("order_code")
        .sort(["buyer_unit", "supplier", "created_date", "order_code"])
        .with_columns(pl.int_range(pl.len()).over(["buyer_unit", "supplier"]).alias("pair_history"))
        .select("order_code", "pair_history")
    )
    return (
        lines.join(first_seen, on="supplier", how="left")
        .join(orders, on="order_code", how="left")
        .with_columns((pl.col("created_date") - pl.col("first_seen")).dt.total_days().alias("supplier_age_days"))
        .drop("first_seen")
    )


def buyer_stats(train: pl.DataFrame, min_lines: int = MIN_BUYER_LINES) -> pl.DataFrame:
    """Median log price each buying unit paid for each product and unit in training. The
    ONU codes are broad ("medical exams", "stationery"), so a price is better judged
    against what the same buyer paid before for the same thing than against everyone."""
    return (
        train.group_by(["buyer_unit", "product_code", "unit"])
        .agg(pl.len().alias("buyer_lines"), pl.col("unit_price").log().median().alias("buyer_price_med"))
        .filter(pl.col("buyer_lines") >= min_lines)
        .drop("buyer_lines")
    )


def group_stats(train: pl.DataFrame, min_lines: int = MIN_GROUP_LINES) -> pl.DataFrame:
    """Median and MAD of log price, log quantity and log total per product and unit."""
    exprs = []
    for col, name in (("unit_price", "price"), ("quantity", "quantity"), ("line_total", "total")):
        log = pl.col(col).log()
        exprs += [log.median().alias(f"{name}_med"), (log - log.median()).abs().median().alias(f"{name}_mad")]
    return (
        train.filter(pl.col("line_total") > 0)
        .group_by(["product_code", "unit"])
        .agg([pl.len().alias("group_lines")] + exprs)
        .filter(pl.col("group_lines") >= min_lines)
    )


def build_features(lines: pl.DataFrame, stats: pl.DataFrame, buyers: pl.DataFrame | None = None) -> pl.DataFrame:
    """Joins the statistics and computes the features; lines whose product and unit are not
    covered by the group statistics are dropped. ``price_vs_buyer`` is the log distance to
    the buyer's own median for the product, 0 when the buyer has no history for it."""
    df = lines.join(stats, on=["product_code", "unit"], how="inner")
    if buyers is not None:
        df = df.join(buyers, on=["buyer_unit", "product_code", "unit"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("buyer_price_med"))
    exprs = []
    for col, name in (("unit_price", "price"), ("quantity", "quantity"), ("line_total", "total")):
        scale = 1.4826 * pl.max_horizontal(pl.col(f"{name}_mad"), pl.lit(MAD_FLOOR))
        exprs.append(((pl.col(col).log() - pl.col(f"{name}_med")) / scale).clip(-Z_CLIP, Z_CLIP).alias(f"{name}_z"))
    exprs += [
        (pl.col("unit_price").log() - pl.col("buyer_price_med")).fill_null(0.0).clip(-Z_CLIP, Z_CLIP).alias("price_vs_buyer"),
        pl.col("supplier_age_days").cast(pl.Float64).log1p().alias("supplier_age_log"),
        pl.col("pair_history").cast(pl.Float64).log1p().alias("pair_history_log"),
    ]
    return df.with_columns(exprs)


# ----------------------------------------------------------------------------- planted anomalies


def plant_anomalies(test: pl.DataFrame, stats: pl.DataFrame, fraction: float = BUDGET, seed: int = SEED) -> pl.DataFrame:
    """Plants ``fraction`` of anomalies, split evenly among ``ANOMALY_TYPES``, into a copy of
    real test lines (raw values only; features are recomputed afterwards). Adds
    ``anomaly`` (null for untouched lines)."""
    rng = np.random.default_rng(seed)
    n = test.height
    chosen = rng.choice(n, size=int(round(n * fraction)), replace=False)
    kinds = np.empty(n, dtype=object)
    kinds[chosen] = np.array(ANOMALY_TYPES)[np.arange(len(chosen)) % len(ANOMALY_TYPES)]

    price = test["unit_price"].to_numpy().astype(float).copy()
    qty = test["quantity"].to_numpy().astype(float).copy()
    age = test["supplier_age_days"].to_numpy().astype(float).copy()
    history = test["pair_history"].to_numpy().astype(float).copy()

    def mask(kind):
        return kinds == kind

    m = mask("overpricing")
    price[m] *= rng.uniform(3, 8, m.sum())
    m = mask("inflated_quantity")
    qty[m] *= rng.uniform(5, 10, m.sum())
    m = mask("digit_error")
    price[m] *= 1000.0  # three extra zeros typed into the unit price
    m = mask("new_supplier")
    price[m] *= rng.uniform(2, 4, m.sum())
    age[m] = 0.0
    history[m] = 0.0
    m = mask("wrong_product")
    if m.any():
        # a price typical of a product at least 10x cheaper or dearer than the line's own
        meds = stats["price_med"].to_numpy()
        own = np.log(price[m])
        far = [np.flatnonzero(np.abs(meds - o) >= np.log(10)) for o in own]
        price[m] = [np.exp(meds[rng.choice(f)]) if len(f) else p * 10 for f, p in zip(far, price[m])]

    return test.with_columns(
        pl.Series("unit_price", price),
        pl.Series("quantity", qty),
        pl.Series("line_total", price * qty),
        pl.Series("supplier_age_days", age),
        pl.Series("pair_history", history),
        pl.Series("anomaly", kinds.tolist(), dtype=pl.String),
    )


# ----------------------------------------------------------------------------- detectors


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std < 1e-8] = 1.0
    return [((x - mean) / std).astype(np.float32) for x in (train, *others)]


def rule_scores(df: pl.DataFrame) -> np.ndarray:
    """What an auditor would do by hand: how far price, quantity and total are from what
    this product usually costs, plus how far the price is from what the same buyer paid
    before, added up. No training."""
    return (df["price_z"].abs() + df["quantity_z"].abs() + df["total_z"].abs() + df["price_vs_buyer"].abs()).to_numpy()


def isolation_forest_scores(X_train: np.ndarray, X_test: np.ndarray, seed: int = SEED) -> np.ndarray:
    model = IsolationForest(n_estimators=300, contamination=BUDGET, random_state=seed, n_jobs=-1)
    model.fit(X_train)
    return -model.score_samples(X_test)  # higher = more anomalous


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, activation: type[nn.Module] = nn.ReLU, bottleneck_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), activation(), nn.Linear(16, 8), activation(), nn.Linear(8, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8), activation(), nn.Linear(8, 16), activation(), nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int


def train_autoencoder(
    X_train: np.ndarray, X_val: np.ndarray, activation: type[nn.Module] = nn.ReLU, seed: int = SEED,
    max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Autoencoder(X_train.shape[1], activation)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.from_numpy(X_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)
        train_losses.append(total / len(X_train))
        model.eval()
        with torch.no_grad():
            val_losses.append(criterion(model(X_val_t), X_val_t).item())
        if val_losses[-1] < best - 1e-6:
            best, best_epoch, stale = val_losses[-1], epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return TrainResult(model, train_losses, val_losses, best_epoch)


def reconstruction_error(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X)
        return torch.mean((model(X_t) - X_t) ** 2, dim=1).numpy()


def recall_at_budget(scores: np.ndarray, anomaly: np.ndarray, budget: float = BUDGET) -> dict:
    """Share of planted anomalies (overall and by type) in the top ``budget`` of scores."""
    anomaly = np.asarray(anomaly, dtype=object)
    k = int(round(len(scores) * budget))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores, kind="stable")[:k]] = True
    planted = anomaly != None  # noqa: E711 -- object array of strings and None
    out = {"overall": float(flagged[planted].mean()), "precision": float(planted[flagged].mean())}
    for kind in ANOMALY_TYPES:
        is_kind = anomaly == kind
        out[kind] = float(flagged[is_kind].mean()) if is_kind.any() else float("nan")
    return out
