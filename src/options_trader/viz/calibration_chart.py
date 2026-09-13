"""
calibration_chart.py — calibration diagnostic for rolling backtests.

Two-panel layout:

    Left  panel — CALIBRATION REGRESSION
        Sorted realised PIT percentiles plotted against uniform quantiles
        (i+0.5)/n. Under perfect calibration the points lie on y=x. A fitted
        OLS line is overlaid; slope ≠ 1 or intercept ≠ 0 signal miscalibration.

    Right panel — PIT HISTOGRAM
        Histogram of raw realised percentiles in 10 bins. Under perfect
        calibration this is uniform (height = 1 on the density scale). U-shape
        means fat-tail miss (too-narrow forecast); inverted-U means
        too-wide forecast.

Log scores (mean ± std-err) are printed in both legends so the chart conveys
calibration AND sharpness simultaneously.

Slope interpretation:
    slope ≈ 1, intercept ≈ 0  → well-calibrated
    slope < 1                 → too wide (over-conservative variance)
    slope > 1                 → too narrow (fat-tail miss, under-conservative)
    intercept ≠ 0             → directional bias
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from options_trader.backtest.log_score import BacktestResult


DEFAULT_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
PIT_HIST_BINS = 10


def _fit_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS y = m*x + b. Returns (slope, intercept)."""
    n = len(x)
    if n < 2:
        return 1.0, 0.0
    A = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def plot_calibration(
    results: Sequence[BacktestResult],
    output_path: str | Path,
    title: str = "Forecast calibration",
) -> Path:
    """Two-panel calibration chart: regression + PIT histogram. Saves PNG."""
    if not results:
        raise ValueError("at least one BacktestResult required")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))

    # ---------- Left: calibration regression ----------
    ax_cal.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2,
                label="Perfect calibration (y = x)")

    for i, result in enumerate(results):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        pcts = np.sort(result.percentiles)
        n = len(pcts)
        uniform_quantiles = (np.arange(n) + 0.5) / n

        slope, intercept = _fit_regression(uniform_quantiles, pcts)
        fit_line = slope * uniform_quantiles + intercept

        legend_text = (
            f"{result.name}\n"
            f"  log score = {result.mean_log_score:.3f} ± {result.log_score_stderr:.3f}\n"
            f"  fit: y = {slope:.3f}·x + {intercept:+.3f}  (n={n})"
        )

        ax_cal.scatter(uniform_quantiles, pcts, color=color, alpha=0.6, s=30,
                       edgecolors="none", label=legend_text)
        ax_cal.plot(uniform_quantiles, fit_line, color=color, linewidth=2.0, alpha=0.9)

    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_aspect("equal")
    ax_cal.set_xlabel("Uniform quantile (expected if calibrated)")
    ax_cal.set_ylabel("Realised PIT percentile (observed)")
    ax_cal.set_title("Calibration regression")
    ax_cal.grid(True, alpha=0.3, linestyle="--")
    ax_cal.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # ---------- Right: PIT histogram ----------
    hist_bins = np.linspace(0, 1, PIT_HIST_BINS + 1)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    bin_width = 1.0 / PIT_HIST_BINS
    n_results = len(results)
    bar_width = (bin_width * 0.9) / n_results   # leave 10% gap between groups

    for i, result in enumerate(results):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        density, _ = np.histogram(result.percentiles, bins=hist_bins, density=True)
        offset = (i - (n_results - 1) / 2) * bar_width

        legend_text = (
            f"{result.name}\n"
            f"  log score = {result.mean_log_score:.3f} ± {result.log_score_stderr:.3f}\n"
            f"  mean PIT = {result.percentiles.mean():.3f}, "
            f"std = {result.percentiles.std():.3f}"
        )
        ax_hist.bar(bin_centers + offset, density,
                    width=bar_width, color=color, alpha=0.75,
                    edgecolor="black", linewidth=0.5, label=legend_text)

    ax_hist.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                    label="Perfect calibration (uniform density = 1)")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Realised PIT percentile")
    ax_hist.set_ylabel("Density (forecasts per unit interval)")
    ax_hist.set_title(f"PIT histogram — flat = calibrated  ({PIT_HIST_BINS} bins)")
    ax_hist.grid(True, alpha=0.3, linestyle="--")
    ax_hist.legend(loc="upper center", fontsize=9, framealpha=0.9)

    fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
