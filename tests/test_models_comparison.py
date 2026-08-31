"""
Tests unitarios para el motor de deteccion de anomalias.

Usa un dataset sintetico pequeno (n=800) generado con la misma funcion que
usa produccion (`generate_procurement_data`) para mantener las pruebas
rapidas, sin mockear el pipeline real de features/entrenamiento.

Ejecutar:
    .\\venv\\Scripts\\python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoencoder import (
    NUMERIC_FEATURES,
    build_feature_matrix,
    compute_category_stats,
    generate_procurement_data,
    reconstruction_error,
    set_seeds,
    train_autoencoder,
)
from models_comparison import (
    ACTIVATIONS,
    ConfigurableAutoencoder,
    baseline_zscore_rule,
    evaluate,
    isolation_forest_scores,
)

SEED = 42
N_SMALL = 800
ANOMALY_FRACTION = 0.05


def _make_small_dataset():
    set_seeds(SEED)
    df = generate_procurement_data(N_SMALL, ANOMALY_FRACTION, SEED)
    idx_train = np.arange(int(N_SMALL * 0.85))
    category_stats = compute_category_stats(df, idx_train)
    X, feature_names = build_feature_matrix(df, category_stats)
    y_true = df["anomalia_sintetica"].to_numpy()
    return df, X, feature_names, y_true, idx_train


def test_generate_procurement_data_anomaly_fraction():
    df = generate_procurement_data(N_SMALL, ANOMALY_FRACTION, SEED)
    n_anomalies = int(df["anomalia_sintetica"].sum())
    assert df.height == N_SMALL
    assert n_anomalies == round(N_SMALL * ANOMALY_FRACTION)
    # las 5 categorias de anomalia inyectada deben aparecer, no "normal" solamente
    kinds = set(df["tipo_anomalia_sintetica"].unique().to_list())
    assert "normal" in kinds
    assert len(kinds) > 1


def test_build_feature_matrix_shape_and_columns():
    _df, X, feature_names, _y, _idx = _make_small_dataset()
    assert X.shape == (N_SMALL, len(NUMERIC_FEATURES))
    assert feature_names == NUMERIC_FEATURES
    assert np.isfinite(X).all(), "las features no deben contener NaN/inf tras el clipping"


def test_baseline_zscore_rule_flags_injected_anomalies_above_chance():
    _df, X, feature_names, y_true, _idx = _make_small_dataset()
    scores = baseline_zscore_rule(X, feature_names)
    assert scores.shape == (N_SMALL,)
    assert (scores >= 0).all()

    result = evaluate("baseline", scores, y_true, ANOMALY_FRACTION)
    # el recall de un flagging aleatorio al 5% seria ~5%; la regla de
    # z-score, incluso en un dataset chico, debe superarlo claramente
    assert result.recall > ANOMALY_FRACTION


def test_isolation_forest_scores_flags_anomalies_above_chance():
    _df, X, _feature_names, y_true, idx_train = _make_small_dataset()
    scores = isolation_forest_scores(X[idx_train], X, SEED)
    assert scores.shape == (N_SMALL,)

    result = evaluate("isolation_forest", scores, y_true, ANOMALY_FRACTION)
    assert result.recall > ANOMALY_FRACTION


def test_evaluate_precision_recall_consistency():
    y_true = np.array([True, True, False, False, False, False, False, False, False, False])
    scores = np.array([10, 9, 8, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    result = evaluate("toy", scores, y_true, anomaly_fraction=0.3)
    # top-30% de 10 filas = 3 filas; 2 de las 3 flageadas son verdaderas anomalias
    assert result.n_flagged == 3
    assert result.true_positives == 2
    assert result.precision == 2 / 3
    assert result.recall == 1.0  # ambas anomalias reales fueron capturadas


def test_configurable_autoencoder_all_activations_train_and_reconstruct():
    """El autoencoder parametrizable debe entrenar y producir errores de
    reconstruccion finitos y no negativos para las 3 activaciones comparadas
    (ReLU/GELU/Swish), sin importar cual se use."""
    _df, X, _feature_names, _y, idx_train = _make_small_dataset()
    idx_val = np.arange(idx_train[-1] + 1, N_SMALL)
    X = X.astype(np.float32)
    X_train, X_val = X[idx_train], X[idx_val]
    device = torch.device("cpu")

    for name, activation_cls in ACTIVATIONS.items():
        set_seeds(SEED)
        model = ConfigurableAutoencoder(input_dim=X.shape[1], activation=activation_cls, bottleneck_dim=2)
        # entrenamiento corto (pocas epochs) solo para validar que el forward
        # pass y el loop de entrenamiento funcionan con cada activacion
        import autoencoder as ae_module

        original_max_epochs = ae_module.MAX_EPOCHS
        original_patience = ae_module.PATIENCE
        ae_module.MAX_EPOCHS = 3
        ae_module.PATIENCE = 3
        try:
            train_autoencoder(model, X_train, X_val, device)
        finally:
            ae_module.MAX_EPOCHS = original_max_epochs
            ae_module.PATIENCE = original_patience

        errors = reconstruction_error(model, X, device)
        assert errors.shape == (N_SMALL,)
        assert np.isfinite(errors).all(), f"errores no finitos con activacion {name}"
        assert (errors >= 0).all(), f"MSE negativo con activacion {name}"
