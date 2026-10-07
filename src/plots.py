"""Figures for the README, written to ``results/figures/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
BLUE, ORANGE, GREY, RED = "#2563eb", "#d97706", "#9ca3af", "#b91c1c"


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bunching(amounts_by_regime: dict[str, np.ndarray], caps: dict[str, float], fits: dict[str, dict]) -> Path:
    """Compra Ágil orders by amount under each rule, with the linear trend the excess is measured against."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    titles = {"cap30": "January-November 2024: cap of 30 UTM", "cap100": "January 2025-September 2026: cap of 100 UTM"}
    for ax, regime in zip(axes, ("cap30", "cap100")):
        a = amounts_by_regime[regime]
        edges = np.arange(0, 101, 1.0)
        counts, _ = np.histogram(a, bins=edges)
        ax.bar(edges[:-1], counts, width=1.0, align="edge", color=GREY, edgecolor="none")
        fit = fits[regime]
        centers, predicted = np.array(fit["centers"]), np.array(fit["predicted"])
        width = centers[1] - centers[0]
        ax.plot(centers, predicted / width, color=BLUE, lw=1.8, label="trend fitted at 60-90% of the cap")
        ax.axvline(caps[regime], color=RED, lw=1.5, ls="--", label=f"cap: {caps[regime]:.0f} UTM")
        ax.set_title(f"{titles[regime]} — {fit['excess_mass']:+.0%} more orders in the last 10% than the trend", fontsize=10)
        ax.set_ylabel("orders per UTM")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25, axis="y")
    axes[-1].set_xlabel("order total including taxes (UTM of the creation month)")
    axes[0].set_ylim(0, None)
    return _save(fig, "bunching_compra_agil.png")


def plot_placebos(real: dict[str, float], placebos: dict[float, float]) -> Path:
    """Excess mass under the real caps against the same measure at caps that never existed."""
    labels = [f"{c:.0f}" for c in placebos] + ["30 (2024,\nreal cap)", "100 (2025-26,\nreal cap)"]
    values = list(placebos.values()) + [real["cap30"], real["cap100"]]
    colors = [GREY] * len(placebos) + [RED, RED]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_ylabel("excess mass in the last 10% below the line")
    ax.set_xlabel("amount treated as a cap (UTM); grey = 2025-26 at amounts with no legal cap")
    ax.set_title("The pile only appears under a real cap, and it moved when the cap moved", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, "placebo_caps.png")


def plot_sibling_rates(rates: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Share of orders with another order of the same buyer and supplier within 7 days, by size."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for regime, color, label in (("cap30", ORANGE, "2024 (cap 30 UTM)"), ("cap100", BLUE, "2025-26 (cap 100 UTM)")):
        size, rate = rates[regime]
        ax.plot(size * 100 + 2.5, rate * 100, marker="o", color=color, label=label)
    ax.set_xlabel("order size, % of the cap")
    ax.set_ylabel("orders with a sibling within 7 days (%)")
    ax.set_title("Orders near the cap come with siblings more often, but mostly because of who buys there", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, "sibling_rates.png")


def plot_detector_recall(recalls: dict[str, dict[str, float]], types: list[str]) -> Path:
    """Share of each kind of planted anomaly found in the top 5% of every detector."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    names = list(recalls)
    width = 0.8 / len(names)
    x = np.arange(len(types) + 1)
    palette = [GREY, BLUE, ORANGE, "#16a34a", "#7c3aed"]
    for i, name in enumerate(names):
        values = [recalls[name][t] for t in types] + [recalls[name]["overall"]]
        ax.bar(x + i * width - 0.4 + width / 2, np.array(values) * 100, width, label=name, color=palette[i % len(palette)])
    ax.axhline(5, color=RED, lw=1, ls="--", label="random review of 5%")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in types] + ["all"])
    ax.set_ylabel("planted anomalies found (%)")
    ax.set_title("Reviewing the top 5% of 200,000 real 2026 lines with 10,000 planted anomalies", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, "detector_recall.png")


def plot_training_curves(curves: dict[str, tuple[list[float], list[float], int]]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for (name, (train, val, best)), color in zip(curves.items(), (BLUE, ORANGE, "#16a34a")):
        epochs = np.arange(1, len(train) + 1)
        ax.plot(epochs, train, color=color, lw=1.2, label=f"{name} (train)")
        ax.plot(epochs, val, color=color, lw=1.2, ls="--", label=f"{name} (validation)")
        ax.axvline(best, color=color, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Autoencoder training on 2025 lines (dotted: epoch kept)", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    return _save(fig, "training_curves.png")
