"""
Motor de deteccion de anomalias en facturacion de proveedores mineros.

Entrena un autoencoder (PyTorch) no supervisado sobre features tabulares de
facturas de compra (monto, cantidad, precio unitario, plazos, categoria,
region de faena) y aisla el 5% de las facturas con mayor error de
reconstruccion como candidatas a auditoria.

El dataset es sintetico (generado con Polars) porque no existe un dataset
publico de facturacion de proveedores mineros chilenos. Se inyecta un 5% de
facturas con patrones de anomalia realistas (sobreprecio, cantidad inflada,
monto no reconciliado, mismatch categoria/precio, proveedor nuevo con monto
alto) cuya etiqueta se guarda SOLO para validar el modelo al final -- nunca
se usa durante el entrenamiento, que es completamente no supervisado.

Uso:
    .\\venv\\Scripts\\python.exe autoencoder.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_INVOICES = 15_000
ANOMALY_FRACTION = 0.05
VAL_FRACTION = 0.15
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 1e-3
BOTTLENECK_DIM = 4

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# cantidad/precio son log-normales y varian en ordenes de magnitud entre
# categorias (combustible ~$850/litro vs neumaticos CAEX ~$8.5M/unidad), asi
# que se expresan como z-score DENTRO de su propia categoria (no en valor
# crudo): un StandardScaler global sobre valores crudos deja que la varianza
# ENTRE categorias ahogue las anomalias DENTRO de una categoria -- verificado
# empiricamente, con features crudas el recall sobre anomalias inyectadas
# caia a ~17% incluso con log1p. monto_ratio_log expone directamente si el
# monto declarado reconcilia con cantidad x precio_unitario (deberia ser ~0
# en una factura normal), que es exactamente lo que manipulan 2 de los 5
# tipos de anomalia inyectada.
NUMERIC_FEATURES = [
    "precio_zscore_categoria",
    "cantidad_zscore_categoria",
    "monto_ratio_log",
    "dias_credito",
    "dias_entrega",
    "dias_como_proveedor_log",
]

# categoria -> (precio_mu_log, precio_sigma_log, cantidad_mu_log, cantidad_sigma_log)
CATEGORY_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "Explosivos": (10.71, 0.30, 5.70, 0.60),
    "Neumaticos CAEX": (15.96, 0.25, 1.20, 0.45),
    "Repuestos Chancado": (13.99, 0.35, 1.80, 0.55),
    "Combustible": (6.75, 0.15, 9.80, 0.50),
    "Lubricantes": (9.39, 0.30, 6.20, 0.55),
    "Servicios Mantencion": (15.07, 0.40, 0.55, 0.40),
    "EPP": (10.13, 0.35, 5.90, 0.60),
    "Repuestos Perforacion": (14.60, 0.35, 1.60, 0.55),
}
CATEGORIES = list(CATEGORY_PARAMS.keys())

REGIONS = [
    "Antofagasta",
    "Atacama",
    "Tarapaca",
    "Valparaiso",
    "O'Higgins",
    "Coquimbo",
]
REGION_WEIGHTS = [0.38, 0.20, 0.14, 0.10, 0.10, 0.08]

PAYMENT_TERMS = [30, 45, 60, 90]
PAYMENT_WEIGHTS = [0.35, 0.30, 0.25, 0.10]

N_PROVEEDORES = 180


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_procurement_data(
    n_invoices: int, anomaly_fraction: float, seed: int
) -> pl.DataFrame:
    """Genera facturas sinteticas de proveedores mineros con un 5% de anomalias inyectadas."""
    rng = np.random.default_rng(seed)

    provider_ids = [f"PROV_{i:04d}" for i in range(N_PROVEEDORES)]
    provider_tenure = rng.integers(15, 3000, size=N_PROVEEDORES)
    provider_region = rng.choice(REGIONS, size=N_PROVEEDORES, p=REGION_WEIGHTS)

    proveedor_idx = rng.integers(0, N_PROVEEDORES, size=n_invoices)
    categoria = rng.choice(CATEGORIES, size=n_invoices)

    cantidad = np.empty(n_invoices, dtype=np.float64)
    precio_unitario = np.empty(n_invoices, dtype=np.float64)
    for cat in CATEGORIES:
        mask = categoria == cat
        p_mu, p_sigma, q_mu, q_sigma = CATEGORY_PARAMS[cat]
        n = int(mask.sum())
        precio_unitario[mask] = rng.lognormal(p_mu, p_sigma, size=n)
        cantidad[mask] = np.ceil(rng.lognormal(q_mu, q_sigma, size=n))

    dias_credito = rng.choice(PAYMENT_TERMS, size=n_invoices, p=PAYMENT_WEIGHTS)
    dias_entrega = np.clip(rng.normal(15, 5, size=n_invoices), 1, 60).round()
    dias_como_proveedor = provider_tenure[proveedor_idx].astype(np.float64)

    monto_declarado = cantidad * precio_unitario * (1 + rng.normal(0, 0.01, size=n_invoices))

    anomaly_type = np.full(n_invoices, "normal", dtype=object)
    n_anomalies = int(round(n_invoices * anomaly_fraction))
    anomaly_idx = rng.choice(n_invoices, size=n_anomalies, replace=False)
    anomaly_kinds = rng.choice(
        ["sobreprecio", "cantidad_inflada", "monto_no_reconciliado", "categoria_precio_mismatch", "proveedor_nuevo_monto_alto"],
        size=n_anomalies,
    )

    for idx, kind in zip(anomaly_idx, anomaly_kinds):
        anomaly_type[idx] = kind
        if kind == "sobreprecio":
            precio_unitario[idx] *= rng.uniform(3.0, 8.0)
            monto_declarado[idx] = cantidad[idx] * precio_unitario[idx]
        elif kind == "cantidad_inflada":
            cantidad[idx] = np.ceil(cantidad[idx] * rng.uniform(5.0, 10.0))
            monto_declarado[idx] = cantidad[idx] * precio_unitario[idx]
        elif kind == "monto_no_reconciliado":
            monto_declarado[idx] = cantidad[idx] * precio_unitario[idx] * rng.uniform(1.4, 2.5)
        elif kind == "categoria_precio_mismatch":
            donor_cat = rng.choice(CATEGORIES)
            p_mu, p_sigma, _, _ = CATEGORY_PARAMS[donor_cat]
            precio_unitario[idx] = rng.lognormal(p_mu, p_sigma)
            monto_declarado[idx] = cantidad[idx] * precio_unitario[idx]
        elif kind == "proveedor_nuevo_monto_alto":
            dias_como_proveedor[idx] = rng.integers(1, 30)
            monto_declarado[idx] = cantidad[idx] * precio_unitario[idx] * rng.uniform(2.0, 4.0)

    base_date = np.datetime64("2024-08-25")
    fecha_emision = base_date + rng.integers(0, 730, size=n_invoices).astype("timedelta64[D]")

    df = pl.DataFrame(
        {
            "invoice_id": [f"INV_{i:06d}" for i in range(n_invoices)],
            "fecha_emision": fecha_emision,
            "proveedor_id": [provider_ids[i] for i in proveedor_idx],
            "region_faena": provider_region[proveedor_idx],
            "categoria_producto": categoria,
            "cantidad": cantidad,
            "precio_unitario_clp": precio_unitario.round(2),
            "monto_declarado_clp": monto_declarado.round(2),
            "dias_credito": dias_credito.astype(np.int64),
            "dias_entrega": dias_entrega.astype(np.int64),
            "dias_como_proveedor": dias_como_proveedor.astype(np.int64),
            "anomalia_sintetica": anomaly_type != "normal",
            "tipo_anomalia_sintetica": anomaly_type,
        }
    )
    return df


def compute_category_stats(df: pl.DataFrame, train_idx: np.ndarray) -> pl.DataFrame:
    """Media/desv. estandar de log-precio y log-cantidad por categoria, usando SOLO el
    split de train (evita leakage). Se probo tambien la version robusta (mediana/MAD)
    para blindar contra la contaminacion no supervisada, pero empiricamente dio peor
    recall global (33.6% vs 37.3%) al re-balancear que tipo de anomalia gana los
    puestos del umbral top-5% -- ver seccion de resultados del README para el detalle
    de este tradeoff estructural entre tipos de anomalia."""
    return (
        df[train_idx]
        .with_columns(
            pl.col("precio_unitario_clp").log1p().alias("_log_precio"),
            pl.col("cantidad").log1p().alias("_log_cantidad"),
        )
        .group_by("categoria_producto")
        .agg(
            precio_log_mean=pl.col("_log_precio").mean(),
            precio_log_std=pl.col("_log_precio").std(),
            cantidad_log_mean=pl.col("_log_cantidad").mean(),
            cantidad_log_std=pl.col("_log_cantidad").std(),
        )
    )


# Winsorizar z-scores/ratios: un mismatch categoria/precio puede generar
# z-scores de decenas de sigmas (precio de neumaticos CAEX evaluado contra la
# distribucion de combustible), y esos outliers extremos dominan el MSE
# durante el entrenamiento y degradan la reconstruccion de todo lo demas --
# mismo patron de bug ya visto y resuelto por winsorizing en el z-score de
# monto de `chile-financial-fraud-detection`. La cota es generosa (no
# recorta el rango real de las anomalias inyectadas, solo sus colas extremas).
ZSCORE_CLIP = 20.0
RATIO_LOG_CLIP = 5.0


def build_feature_matrix(
    df: pl.DataFrame, category_stats: pl.DataFrame
) -> tuple[np.ndarray, list[str]]:
    engineered = (
        df.join(category_stats, on="categoria_producto", how="left")
        .with_columns(
            (
                (pl.col("precio_unitario_clp").log1p() - pl.col("precio_log_mean"))
                / pl.col("precio_log_std")
            ).clip(-ZSCORE_CLIP, ZSCORE_CLIP).alias("precio_zscore_categoria"),
            (
                (pl.col("cantidad").log1p() - pl.col("cantidad_log_mean"))
                / pl.col("cantidad_log_std")
            ).clip(-ZSCORE_CLIP, ZSCORE_CLIP).alias("cantidad_zscore_categoria"),
            (
                pl.col("monto_declarado_clp") / (pl.col("cantidad") * pl.col("precio_unitario_clp"))
            ).log().clip(-RATIO_LOG_CLIP, RATIO_LOG_CLIP).alias("monto_ratio_log"),
            pl.col("dias_como_proveedor").log1p().alias("dias_como_proveedor_log"),
        )
    )
    # region_faena/categoria_producto no se incluyen como one-hot: ningun tipo
    # de anomalia inyectada toca esas etiquetas directamente (la info de
    # categoria ya esta incorporada via z-score relativo a su categoria), y
    # 14 columnas one-hot que el autoencoder reconstruye casi perfectamente
    # para TODAS las filas solo diluyen el MSE promedio con "relleno" sin
    # señal -- verificado empiricamente: incluirlas bajaba el recall.
    features = engineered.select(NUMERIC_FEATURES)
    feature_names = features.columns
    return features.to_numpy().astype(np.float32), feature_names


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = BOTTLENECK_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class TrainResult:
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int


def train_autoencoder(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
) -> TrainResult:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)), batch_size=BATCH_SIZE, shuffle=True
    )
    X_val_t = torch.from_numpy(X_val).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        train_loss = epoch_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), X_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d}/{MAX_EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if patience_counter >= PATIENCE:
            print(f"  early stopping en epoch {epoch} (mejor epoch: {best_epoch}, val_loss={best_val_loss:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(train_losses, val_losses, best_epoch)


def save_training_curve_animation(
    train_losses: list[float], val_losses: list[float], path: Path, n_frames: int = 40
) -> None:
    """Racing line-chart GIF of the (real, already-computed) train/val loss curves."""
    n_epochs = len(train_losses)
    n_frames = min(n_frames, n_epochs)
    frame_epochs = sorted(set(np.linspace(1, n_epochs, n_frames, dtype=int)))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        (train_line,) = ax.plot([], [], color="#4C72B0", lw=2, label="Train loss")
        (val_line,) = ax.plot([], [], color="#DD8452", lw=2, label="Val loss")
        train_label = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#4C72B0", ec="none", alpha=0.9),
        )
        val_label = ax.annotate(
            "", xy=(0, 0), xytext=(10, -20), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#DD8452", ec="none", alpha=0.9),
        )
        ax.set_xlim(0, n_epochs)
        y_max = max(max(train_losses), max(val_losses)) * 1.1
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title("Curva de entrenamiento del autoencoder (animada)")
        ax.legend(loc="upper right")
        fig.tight_layout()

        def update(frame_idx: int):
            epoch = frame_epochs[frame_idx]
            x = list(range(1, epoch + 1))
            train_line.set_data(x, train_losses[:epoch])
            val_line.set_data(x, val_losses[:epoch])
            train_label.xy = (epoch, train_losses[epoch - 1])
            train_label.set_text(f"Train loss: {train_losses[epoch - 1]:.4f}")
            val_label.xy = (epoch, val_losses[epoch - 1])
            val_label.set_text(f"Val loss: {val_losses[epoch - 1]:.4f}")
            return train_line, val_line, train_label, val_label

        ani = FuncAnimation(fig, update, frames=len(frame_epochs), interval=150, blit=False)
        ani.save(path, writer="pillow")
        plt.close(fig)


def reconstruction_error(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        reconstruction = model(X_t)
        error = torch.mean((reconstruction - X_t) ** 2, dim=1)
    return error.cpu().numpy()


def main() -> None:
    set_seeds(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print(f"\nGenerando {N_INVOICES:,} facturas sinteticas (semilla={SEED})...")
    df = generate_procurement_data(N_INVOICES, ANOMALY_FRACTION, SEED)
    n_true_anomalies = int(df["anomalia_sintetica"].sum())
    print(f"  {n_true_anomalies} facturas con anomalia inyectada ({n_true_anomalies / N_INVOICES:.1%})")
    df.write_csv(DATA_DIR / "procurement_invoices.csv")

    idx_train, idx_val = train_test_split(
        np.arange(N_INVOICES), test_size=VAL_FRACTION, random_state=SEED, shuffle=True
    )

    category_stats = compute_category_stats(df, idx_train)
    X, feature_names = build_feature_matrix(df, category_stats)
    print(f"  matriz de features: {X.shape[0]} filas x {X.shape[1]} columnas -> {feature_names}")

    scaler = StandardScaler()
    n_numeric = len(NUMERIC_FEATURES)
    scaler.fit(X[idx_train][:, :n_numeric])
    X_scaled = X.copy()
    X_scaled[:, :n_numeric] = scaler.transform(X[:, :n_numeric]).astype(np.float32)

    X_train, X_val = X_scaled[idx_train], X_scaled[idx_val]

    print(f"\nEntrenando autoencoder (train={len(X_train)}, val={len(X_val)})...")
    model = Autoencoder(input_dim=X_scaled.shape[1]).to(device)
    result = train_autoencoder(model, X_train, X_val, device)

    print("\nCalculando error de reconstruccion sobre el dataset completo...")
    errors = reconstruction_error(model, X_scaled, device)
    threshold = float(np.percentile(errors, 100 * (1 - ANOMALY_FRACTION)))
    flagged = errors >= threshold

    n_flagged = int(flagged.sum())
    true_positives = int((flagged & df["anomalia_sintetica"].to_numpy()).sum())
    precision = true_positives / n_flagged if n_flagged else 0.0
    recall = true_positives / n_true_anomalies if n_true_anomalies else 0.0

    print(f"\nUmbral (percentil 95 del error de reconstruccion): {threshold:.5f}")
    print(f"Facturas aisladas como anomalas: {n_flagged} ({n_flagged / N_INVOICES:.1%})")
    print(f"Validacion contra anomalias sinteticas inyectadas (solo para reporte, no usada en entrenamiento):")
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   TP: {true_positives}/{n_true_anomalies}")

    recall_by_type = (
        df.with_columns(pl.Series("flagged", flagged), pl.Series("error", errors))
        .filter(pl.col("anomalia_sintetica"))
        .group_by("tipo_anomalia_sintetica")
        .agg(
            total=pl.len(),
            detectadas=pl.col("flagged").sum(),
        )
        .with_columns((pl.col("detectadas") / pl.col("total")).alias("recall"))
        .sort("tipo_anomalia_sintetica")
    )
    print("\nRecall por tipo de anomalia inyectada:")
    for row in recall_by_type.iter_rows(named=True):
        print(f"  {row['tipo_anomalia_sintetica']:<28} {row['detectadas']:>3}/{row['total']:<3}  recall={row['recall']:.2f}")

    results_df = (
        df.with_columns(pl.Series("reconstruction_error", errors))
        .filter(pl.Series(flagged))
        .sort("reconstruction_error", descending=True)
    )
    results_df.write_csv(RESULTS_DIR / "anomalias_detectadas.csv")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_scaled.shape[1],
            "bottleneck_dim": BOTTLENECK_DIM,
            "feature_names": feature_names,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "threshold": threshold,
        },
        MODELS_DIR / "autoencoder.pt",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.train_losses, label="Train loss")
    ax.plot(result.val_losses, label="Val loss")
    ax.axvline(result.best_epoch - 1, color="gray", linestyle="--", alpha=0.6, label=f"Mejor epoch ({result.best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Curva de entrenamiento del autoencoder")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curve.png", dpi=150)
    plt.close(fig)

    save_training_curve_animation(
        result.train_losses, result.val_losses, RESULTS_DIR / "training_curve_animated.gif"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=80, color="#4C72B0", alpha=0.8)
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"Umbral P95 ({threshold:.4f})")
    ax.set_xlabel("Error de reconstruccion (MSE)")
    ax.set_ylabel("Cantidad de facturas")
    ax.set_title("Distribucion del error de reconstruccion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "reconstruction_error_hist.png", dpi=150)
    plt.close(fig)

    print(f"\nArtefactos guardados en {DATA_DIR}, {MODELS_DIR} y {RESULTS_DIR}")


if __name__ == "__main__":
    main()
