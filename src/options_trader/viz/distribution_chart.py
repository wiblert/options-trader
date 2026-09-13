"""
distribution_chart.py — plot weighted empirical distributions of returns.

Renders one or more ReturnDistributions as overlaid weighted histograms.
The y-axis is density (so distributions with different sample counts are
comparable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from options_trader.forecast.return_distribution import ReturnDistribution


# Sensible default colors for up to 4 overlaid distributions.
DEFAULT_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


def _weighted_stats(rd: ReturnDistribution) -> tuple[float, float, float, float]:
    """Return (weighted mean, weighted std, weighted 5pct, weighted 95pct)."""
    mean = float(np.sum(rd.samples * rd.weights))
    var = float(np.sum(rd.weights * (rd.samples - mean) ** 2))
    # Weighted quantiles via cumulative weights on sorted samples
    order = np.argsort(rd.samples)
    s_sorted = rd.samples[order]
    w_cum = np.cumsum(rd.weights[order])
    q05 = float(np.interp(0.05, w_cum, s_sorted))
    q95 = float(np.interp(0.95, w_cum, s_sorted))
    return mean, float(np.sqrt(var)), q05, q95


def plot_return_distribution(
    distributions: ReturnDistribution | Mapping[str, ReturnDistribution],
    output_path: str | Path,
    title: str = "Return distribution",
    bins: int = 60,
    clip_quantiles: tuple[float, float] = (0.001, 0.999),
) -> Path:
    """Plot one or more weighted return distributions as overlaid histograms.

    Args:
        distributions: A single ReturnDistribution or a mapping of label ->
            ReturnDistribution. Mappings produce overlaid histograms for
            comparison.
        output_path: PNG destination. Parent directory is created if missing.
        title: Plot title.
        bins: Histogram bin count (shared across all distributions).
        clip_quantiles: x-axis range clipped to these weighted quantiles of the
            union of all samples — keeps extreme outliers from squashing the body.

    Returns:
        Absolute Path of the saved PNG.
    """
    if isinstance(distributions, ReturnDistribution):
        distributions = {"distribution": distributions}
    if len(distributions) == 0:
        raise ValueError("at least one distribution required")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine shared x-axis range from union of samples
    all_samples = np.concatenate([rd.samples for rd in distributions.values()])
    all_weights = np.concatenate([rd.weights for rd in distributions.values()])
    all_weights = all_weights / all_weights.sum()
    order = np.argsort(all_samples)
    cum = np.cumsum(all_weights[order])
    lo = float(np.interp(clip_quantiles[0], cum, all_samples[order]))
    hi = float(np.interp(clip_quantiles[1], cum, all_samples[order]))
    bin_edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)

    for i, (label, rd) in enumerate(distributions.items()):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        mean, std, q05, q95 = _weighted_stats(rd)
        legend = (
            f"{label}\n"
            f"  n={len(rd)}, mean={mean:.4f}, std={std:.4f}\n"
            f"  5%={q05:.4f}, 95%={q95:.4f}"
        )
        ax.hist(
            rd.samples,
            bins=bin_edges,
            weights=rd.weights,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            edgecolor=color,
            facecolor=color,
            linewidth=1.2,
            label=legend,
        )

    ax.set_xlabel("Daily log-return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
