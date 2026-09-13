"""
block_bootstrap_forecaster.py — moving-block (K-day) Monte Carlo forecaster.

The plain BootstrapForecaster draws horizon_days *independent* daily log-returns
per path and sums them. That sum assumes daily log-returns are iid: it shuffles
away any serial dependence (volatility clustering, momentum, mean reversion) the
real series had. The K-day log-return it produces has the right marginal variance
(√K scaling) but the wrong dynamics.

This forecaster instead draws ONE contiguous K-day block per path, where K equals
the requested horizon. A block is the sum of K *consecutive* daily log-returns,
log(close[i+K] / close[i]) — so whatever serial structure occurred over those K
days is preserved exactly. Comparing log scores against BootstrapForecaster on the
same ReturnDistribution isolates a single question: does the iid-ness of daily
log-returns matter for our horizon?

Construction of blocks (overlapping / "moving block"):
    Given time-ordered daily samples r_0 .. r_{n-1} (oldest -> newest) and block
    length K, the overlapping blocks are
        B_j = sum(r_j .. r_{j+K-1})   for j = 0 .. n-K
    giving n-K+1 blocks. Overlapping windows maximise the number of blocks
    (consecutive blocks share K-1 days — harmless here, since we draw from them
    rather than do inference on them).

Block weighting (anchor on block end):
    Each block inherits the weight of its NEWEST constituent day, i.e. the daily
    weight at index j+K-1. This keeps the exponential-decay recency semantics of
    the daily ReturnDistribution: blocks whose regime ended recently weigh more.

IMPORTANT — do not pass a smoothed distribution:
    This forecaster relies on `distribution.samples` being a pure time series
    (contiguity == temporal adjacency). ReturnDistribution.smooth_samples()
    appends jittered copies that destroy time ordering, which would make blocks
    span non-adjacent days. Build blocks from the raw (optionally decay-weighted)
    distribution only.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from options_trader.forecast.base import Forecaster
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


logger = logging.getLogger(__name__)


DEFAULT_N_PATHS = 10_000


class BlockBootstrapForecaster(Forecaster):
    """Moving-block bootstrap: draws contiguous K-day return blocks (K = horizon).

    Fits the same factory signature as BootstrapForecaster (takes a daily
    ReturnDistribution), so the two can be run head-to-head through the existing
    rolling_log_score_backtest with no harness changes.

    Configuration (constructor):
        distribution: ReturnDistribution of DAILY log-returns, time-ordered
            oldest -> newest (the canonical ReturnDistribution layout). Must not
            be a smoothed distribution (see module docstring).
        n_paths: number of Monte Carlo paths per forecast call (default 10,000).
        seed: optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        distribution: ReturnDistribution,
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
    ) -> None:
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        self.distribution = distribution
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        """Forecast terminal price distribution at horizon_days from spot.

        The block length K is set equal to horizon_days: each path draws one
        contiguous K-day block, so no summing across draws is performed.
        """
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")

        samples = self.distribution.samples       # time-ordered oldest -> newest
        weights = self.distribution.weights        # parallel daily weights, sum 1
        n = len(samples)
        K = horizon_days
        if n < K:
            raise ValueError(
                f"need at least horizon_days={K} daily samples to form one block, "
                f"got {n}"
            )

        # Overlapping K-day block sums via a cumulative-sum difference:
        #   block_sums[j] = sum(samples[j : j+K]) = csum[j+K] - csum[j]
        csum = np.concatenate(([0.0], np.cumsum(samples)))
        block_sums = csum[K:] - csum[:-K]          # length n-K+1

        # Anchor each block's weight on its newest day (index j+K-1). Renormalise
        # over blocks to get sampling probabilities; raw-vs-normalised daily
        # weights give identical probabilities after this step.
        block_w = weights[K - 1:]                   # length n-K+1
        block_p = block_w / block_w.sum()

        chosen = self.rng.choice(len(block_sums), size=self.n_paths, p=block_p)
        k_day_log_returns = block_sums[chosen]
        terminal_prices = spot * np.exp(k_day_log_returns)

        logger.debug(
            "BlockBootstrapForecaster: n_paths=%d K=%d n_blocks=%d spot=%.4f -> "
            "mean=%.4f std=%.4f",
            self.n_paths, K, len(block_sums), spot,
            float(terminal_prices.mean()), float(terminal_prices.std()),
        )
        return PriceDistribution(prices=terminal_prices)
