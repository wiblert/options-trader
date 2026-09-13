"""
price_distribution_chart.py — plot weighted empirical distributions over prices.

Mirror of distribution_chart.py but for PriceDistribution objects. Supports
optional vertical-line annotations for spot (training endpoint) and realized
(out-of-sample truth), and prints the percentile of the realized value within
the forecast PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from options_trader.forecast.price_distribution import PriceDistribution


DEFAULT_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


def _percentile_of(pd: PriceDistribution, value: float) -> float:
    """Weighted CDF of the PriceDistribution evaluated at `value`."""
    order = np.argsort(pd.prices)
    return float(np.interp(value, pd.prices[order], np.cumsum(pd.weights[order])))


def plot_price_distribution(
    distributions: PriceDistribution | Mapping[str, PriceDistribution],
    output_path: str | Path,
    title: str = "Forecast price distribution",
    spot: Optional[float] = None,
    realized: Optional[float] = None,
    bins: int = 60,
    clip_quantiles: tuple[float, float] = (0.001, 0.999),
) -> Path:
    """Plot one or more PriceDistributions as overlaid weighted histograms.

    Args:
        distributions: A PriceDistribution or a mapping of label -> PriceDistribution.
        output_path: PNG destination.
        title: Plot title.
        spot: Optional vertical line at the training-endpoint price (start of forecast).
        realized: Optional vertical line at the out-of-sample realized price.
            When provided alongside a single distribution, the realized's percentile
            within that distribution is annotated on the plot.
        bins: Histogram bin count.
        clip_quantiles: x-axis trimmed to weighted quantiles of the union.
    """
    if isinstance(distributions, PriceDistribution):
        distributions = {"forecast": distributions}
    if len(distributions) == 0:
        raise ValueError("at least one distribution required")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_prices = np.concatenate([pd.prices for pd in distributions.values()])
    all_weights = np.concatenate([pd.weights for pd in distributions.values()])
    all_weights = all_weights / all_weights.sum()
    order = np.argsort(all_prices)
    cum = np.cumsum(all_weights[order])
    lo = float(np.interp(clip_quantiles[0], cum, all_prices[order]))
    hi = float(np.interp(clip_quantiles[1], cum, all_prices[order]))
    # Make sure spot and realized are visible even if they fall outside the clip range
    candidates = [lo, hi]
    if spot is not None:
        candidates.append(float(spot))
    if realized is not None:
        candidates.append(float(realized))
    lo, hi = min(candidates), max(candidates)
    bin_edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, pd) in enumerate(distributions.items()):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        legend = (
            f"{label}\n"
            f"  n={len(pd)}, mean={pd.mean():.2f}, std={pd.std():.2f}\n"
            f"  5%={pd.quantile(0.05):.2f}, 95%={pd.quantile(0.95):.2f}"
        )
        ax.hist(
            pd.prices,
            bins=bin_edges,
            weights=pd.weights,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            edgecolor=color,
            facecolor=color,
            linewidth=1.2,
            label=legend,
        )

    if spot is not None:
        ax.axvline(spot, color="black", linewidth=1.2, linestyle="--",
                   label=f"Spot (train end) = {spot:.2f}")

    if realized is not None:
        ax.axvline(realized, color="#e6550d", linewidth=2.0,
                   label=f"Realized = {realized:.2f}")
        if len(distributions) == 1:
            (pd_single,) = distributions.values()
            pct = _percentile_of(pd_single, realized) * 100
            ax.text(
                0.99, 0.97,
                f"Realized percentile in forecast: {pct:.1f}%",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=11,
                bbox=dict(facecolor="white", edgecolor="#e6550d", alpha=0.9),
            )

    ax.set_xlabel("Terminal price ($)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
