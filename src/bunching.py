"""Bunching below a cap: how many more orders sit just under it than a smooth trend predicts.

A legal cap leaves no orders above it, so the counterfactual can only come from below: the
order counts in bins from ``FIT_FROM`` to the start of the window are fitted with a
low-degree polynomial (a straight line by default) and extrapolated into the window just
under the cap. In the data the density is flat between 60% and 90% of either cap and only
starts to climb after that; a fifth-degree fit extrapolated one-sided swung from +0.65 to
-7.5 with the degree, so the default stays deliberately simple. The excess mass ``b`` is
observed minus predicted orders in the window, divided by the predicted: ``b = 1.0`` means
twice as many orders as the trend implies. The interval comes from refitting on Poisson
resamples of the bin counts.

Bunching under a cap is consistent with two different behaviours that the histogram alone
cannot tell apart: buyers sizing a purchase to the maximum allowed, and purchases split to
stay under it (*fraccionamiento*, which the law forbids). `src.splitting` looks for the
second one directly.
"""

from __future__ import annotations

import numpy as np

BINS_PER_CAP = 100  # bin width = cap / 100: 1 UTM under a 100 UTM cap, 0.3 under 30
WINDOW = 0.10  # the last 10% under the cap
FIT_FROM = 0.6  # the fit uses bins from 60% of the cap up to the window
DEGREE = 1
N_BOOT = 500


def bin_counts(amounts: np.ndarray, cap: float, bins_per_cap: int = BINS_PER_CAP, start: float = FIT_FROM):
    """Counts of ``amounts`` in equal bins from ``start * cap`` up to the cap, the cap itself
    included in the last bin (the law allows orders of exactly the cap)."""
    edges = np.linspace(start * cap, cap, int(round((1 - start) * bins_per_cap)) + 1)
    counts, _ = np.histogram(np.asarray(amounts, dtype=float), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def _excess(centers: np.ndarray, counts: np.ndarray, cap: float, window: float, degree: int) -> tuple[float, np.ndarray]:
    in_window = centers >= cap * (1 - window)
    x = centers / cap  # rescaled so the polynomial is well conditioned
    coefs = np.polyfit(x[~in_window], counts[~in_window], degree)
    predicted = np.polyval(coefs, x)
    expected = predicted[in_window].sum()
    return float((counts[in_window].sum() - expected) / expected), predicted


def bunching(
    amounts: np.ndarray,
    cap: float,
    window: float = WINDOW,
    degree: int = DEGREE,
    fit_from: float = FIT_FROM,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict:
    centers, counts = bin_counts(amounts, cap, start=fit_from)
    b, predicted = _excess(centers, counts, cap, window, degree)
    rng = np.random.default_rng(seed)
    boot = [_excess(centers, rng.poisson(counts), cap, window, degree)[0] for _ in range(n_boot)]
    in_window = centers >= cap * (1 - window)
    return {
        "cap": cap,
        "excess_mass": b,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "observed_in_window": int(counts[in_window].sum()),
        "expected_in_window": float(predicted[in_window].sum()),
        "orders_in_range": int(counts.sum()),
        "centers": centers.tolist(),
        "counts": counts.tolist(),
        "predicted": predicted.tolist(),
    }


def sensitivity(
    amounts: np.ndarray, cap: float, degrees=(0, 1, 2), windows=(0.05, 0.10, 0.15), fit_froms=(0.5, 0.6, 0.7)
) -> list[dict]:
    """The excess mass under every combination of degree, window width and fit range."""
    out = []
    for fit_from in fit_froms:
        centers, counts = bin_counts(amounts, cap, start=fit_from)
        for window in windows:
            for degree in degrees:
                b, _ = _excess(centers, counts, cap, window, degree)
                out.append({"fit_from": fit_from, "window": window, "degree": degree, "excess_mass": b})
    return out
