"""
Comparacion de 3 enfoques complementarios de deteccion de anomalias en
facturacion de proveedores mineros, sobre el mismo dataset sintetico y las
mismas features que usa ``autoencoder.py`` (import directo, sin duplicar
generacion de datos ni feature engineering):

1. Baseline interpretable: regla de z-score combinado (score = suma de
   |z| de precio, |z| de cantidad y |monto_ratio_log|, sin caja negra).
2. Isolation Forest (ensamble de arboles, scikit-learn) sobre las mismas
   features tabulares.
3. Autoencoder PyTorch (reutilizando la arquitectura de autoencoder.py) con
   3 funciones de activacion comparadas: ReLU, GELU y SiLU/Swish.

Los 3 enfoques se evaluan con el mismo protocolo que autoencoder.py: aislar
el 5% de facturas con mayor score de anomalia y medir precision/recall
contra las anomalias sinteticas inyectadas (la etiqueta nunca se usa para
entrenar, solo para reportar). Metricas y predicciones se persisten en
DuckDB (``results/metrics.duckdb``) para poder consultarlas con SQL.

Uso:
    .\\venv\\Scripts\\python.exe models_comparison.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from autoencoder import (
    ANOMALY_FRACTION,
    DATA_DIR,
    MODELS_DIR,
    N_INVOICES,
    NUMERIC_FEATURES,
    RESULTS_DIR,
    ROOT,
    SEED,
    VAL_FRACTION,
    build_feature_matrix,
    compute_category_stats,
    generate_procurement_data,
    reconstruction_error,
    set_seeds,
    train_autoencoder,
)

DUCKDB_PATH = ROOT / "results" / "metrics.duckdb"

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Swish (SiLU)": nn.SiLU,
}


@dataclass
class ApproachResult:
    name: str
    scores: np.ndarray
    flagged: np.ndarray
    precision: float
    recall: float
    n_flagged: int
    true_positives: int


def evaluate(name: str, scores: np.ndarray, y_true: np.ndarray, anomaly_fraction: float) -> ApproachResult:
    threshold = float(np.percentile(scores, 100 * (1 - anomaly_fraction)))
    flagged = scores >= threshold
    n_flagged = int(flagged.sum())
    n_true = int(y_true.sum())
    tp = int((flagged & y_true).sum())
    precision = tp / n_flagged if n_flagged else 0.0
    recall = tp / n_true if n_true else 0.0
    return ApproachResult(name, scores, flagged, precision, recall, n_flagged, tp)


# --- 1. Baseline interpretable: regla de z-score combinado -----------------
# Sin entrenamiento ni caja negra: cada factura recibe un score = suma de la
# magnitud de sus z-scores de precio/cantidad (relativos a su categoria) mas
# el |monto_ratio_log|. Es exactamente la logica que un analista de
# auditoria aplicaria a mano con una planilla -- sirve de piso de referencia
# interpretable para justificar la complejidad de los otros dos enfoques.
def baseline_zscore_rule(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    idx_precio = feature_names.index("precio_zscore_categoria")
    idx_cantidad = feature_names.index("cantidad_zscore_categoria")
    idx_ratio = feature_names.index("monto_ratio_log")
    return np.abs(X[:, idx_precio]) + np.abs(X[:, idx_cantidad]) + np.abs(X[:, idx_ratio])


# --- 2. Isolation Forest -----------------------------------------------
def isolation_forest_scores(X_train: np.ndarray, X_full: np.ndarray, seed: int) -> np.ndarray:
    model = IsolationForest(
        n_estimators=300,
        contamination=ANOMALY_FRACTION,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    # score_samples: mayor = mas normal -> invertimos el signo para que,
    # igual que en los otros dos enfoques, "mayor score = mas anomalo".
    return -model.score_samples(X_full)


# --- 3. Autoencoder con activacion parametrizable -----------------------
class ConfigurableAutoencoder(nn.Module):
    def __init__(self, input_dim: int, activation: type[nn.Module], bottleneck_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            activation(),
            nn.Linear(16, 8),
            activation(),
            nn.Linear(8, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            activation(),
            nn.Linear(8, 16),
            activation(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def run_activation_comparison(
    X_train: np.ndarray, X_val: np.ndarray, X_full: np.ndarray, device: torch.device, seed: int
) -> dict[str, tuple[np.ndarray, list[float], list[float]]]:
    """Entrena un autoencoder identico (misma arquitectura, misma data, mismo
    seed) por cada funcion de activacion y devuelve error de reconstruccion +
    curvas de loss, para poder comparar tanto deteccion como convergencia."""
    out = {}
    for name, activation_cls in ACTIVATIONS.items():
        print(f"\n  -- activacion: {name} --")
        set_seeds(seed)
        model = ConfigurableAutoencoder(input_dim=X_train.shape[1], activation=activation_cls).to(device)
        result = train_autoencoder(model, X_train, X_val, device)
        errors = reconstruction_error(model, X_full, device)
        out[name] = (errors, result.train_losses, result.val_losses)
    return out


def persist_to_duckdb(
    approach_metrics: list[dict],
    predictions_df: pl.DataFrame,
) -> None:
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS approach_metrics (
            run_ts TIMESTAMP DEFAULT current_timestamp,
            approach VARCHAR,
            precision DOUBLE,
            recall DOUBLE,
            n_flagged INTEGER,
            true_positives INTEGER,
            n_true_anomalies INTEGER
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            run_ts TIMESTAMP DEFAULT current_timestamp,
            invoice_id VARCHAR,
            approach VARCHAR,
            score DOUBLE,
            flagged BOOLEAN,
            anomalia_sintetica BOOLEAN
        )
        """
    )
    con.executemany(
        "INSERT INTO approach_metrics (approach, precision, recall, n_flagged, true_positives, n_true_anomalies) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (m["approach"], m["precision"], m["recall"], m["n_flagged"], m["true_positives"], m["n_true_anomalies"])
            for m in approach_metrics
        ],
    )
    con.register("predictions_df", predictions_df)
    con.execute(
        "INSERT INTO predictions (invoice_id, approach, score, flagged, anomalia_sintetica) "
        "SELECT invoice_id, approach, score, flagged, anomalia_sintetica FROM predictions_df"
    )
    con.close()
    print(f"\nMetricas y predicciones persistidas en {DUCKDB_PATH}")


def plot_model_comparison(results: list[ApproachResult], path: Path) -> None:
    names = [r.name for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, precisions, width, label="Precision", color="#4C72B0")
    ax.bar(x + width / 2, recalls, width, label="Recall", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Comparacion de enfoques @ 5% del volumen auditado")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_activation_comparison(
    activation_results: dict[str, tuple[np.ndarray, list[float], list[float]]], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"ReLU": "#4C72B0", "GELU": "#55A868", "Swish (SiLU)": "#C44E52"}
    for name, (_errors, _train, val_losses) in activation_results.items():
        ax.plot(val_losses, label=name, color=colors.get(name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss (MSE)")
    ax.set_title("Curva de validacion por funcion de activacion (autoencoder)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    set_seeds(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    csv_path = DATA_DIR / "procurement_invoices.csv"
    if csv_path.exists():
        print(f"\nCargando facturas existentes desde {csv_path}...")
        df = pl.read_csv(csv_path, try_parse_dates=True)
    else:
        print(f"\nGenerando {N_INVOICES:,} facturas sinteticas (semilla={SEED})...")
        df = generate_procurement_data(N_INVOICES, ANOMALY_FRACTION, SEED)
        df.write_csv(csv_path)

    n_invoices = df.height
    y_true = df["anomalia_sintetica"].to_numpy()
    n_true_anomalies = int(y_true.sum())

    idx_train, idx_val = train_test_split(
        np.arange(n_invoices), test_size=VAL_FRACTION, random_state=SEED, shuffle=True
    )

    category_stats = compute_category_stats(df, idx_train)
    X, feature_names = build_feature_matrix(df, category_stats)

    scaler = StandardScaler()
    n_numeric = len(NUMERIC_FEATURES)
    scaler.fit(X[idx_train][:, :n_numeric])
    X_scaled = X.copy()
    X_scaled[:, :n_numeric] = scaler.transform(X[:, :n_numeric]).astype(np.float32)
    X_train, X_val = X_scaled[idx_train], X_scaled[idx_val]

    results: list[ApproachResult] = []
    prediction_frames: list[pl.DataFrame] = []

    print("\n[1/3] Baseline: regla de z-score combinado...")
    baseline_scores = baseline_zscore_rule(X, feature_names)
    r1 = evaluate("Baseline (z-score)", baseline_scores, y_true, ANOMALY_FRACTION)
    results.append(r1)
    print(f"  Precision={r1.precision:.3f}  Recall={r1.recall:.3f}  TP={r1.true_positives}/{n_true_anomalies}")

    print("\n[2/3] Isolation Forest...")
    if_scores = isolation_forest_scores(X_train, X_scaled, SEED)
    r2 = evaluate("Isolation Forest", if_scores, y_true, ANOMALY_FRACTION)
    results.append(r2)
    print(f"  Precision={r2.precision:.3f}  Recall={r2.recall:.3f}  TP={r2.true_positives}/{n_true_anomalies}")

    print("\n[3/3] Autoencoder PyTorch: comparacion de activaciones ReLU / GELU / Swish...")
    activation_results = run_activation_comparison(X_train, X_val, X_scaled, device, SEED)
    best_activation = None
    best_recall = -1.0
    for name, (errors, _train_losses, _val_losses) in activation_results.items():
        r = evaluate(f"Autoencoder ({name})", errors, y_true, ANOMALY_FRACTION)
        results.append(r)
        print(f"  {name:<14} Precision={r.precision:.3f}  Recall={r.recall:.3f}  TP={r.true_positives}/{n_true_anomalies}")
        if r.recall > best_recall:
            best_recall = r.recall
            best_activation = name

    print(f"\nMejor activacion por recall: {best_activation} ({best_recall:.3f})")

    for r in results:
        prediction_frames.append(
            pl.DataFrame(
                {
                    "invoice_id": df["invoice_id"],
                    "approach": [r.name] * n_invoices,
                    "score": r.scores.astype(np.float64),
                    "flagged": r.flagged,
                    "anomalia_sintetica": y_true,
                }
            )
        )
    predictions_df = pl.concat(prediction_frames)

    approach_metrics = [
        {
            "approach": r.name,
            "precision": r.precision,
            "recall": r.recall,
            "n_flagged": r.n_flagged,
            "true_positives": r.true_positives,
            "n_true_anomalies": n_true_anomalies,
        }
        for r in results
    ]
    persist_to_duckdb(approach_metrics, predictions_df)

    plot_model_comparison(results, RESULTS_DIR / "model_comparison.png")
    plot_activation_comparison(activation_results, RESULTS_DIR / "activation_comparison.png")
    print(f"\nGraficos guardados en {RESULTS_DIR}")

    print("\nResumen final (para README):")
    print(f"{'Enfoque':<26}{'Precision':>12}{'Recall':>12}")
    for r in results:
        print(f"{r.name:<26}{r.precision:>12.3f}{r.recall:>12.3f}")


if __name__ == "__main__":
    main()
