"""
bootstrap_forecaster.py — Monte Carlo bootstrap forecaster.

For each of n_paths Monte Carlo paths, draws horizon_days iid log-returns from
the supplied ReturnDistribution, sums them (log-return additivity gives the
K-day log-return), then applies S_T = spot * exp(R) to produce a
PriceDistribution of n_paths terminal prices with uniform weights.
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


class BootstrapForecaster(Forecaster):
    """Monte Carlo forecaster driven by ReturnDistribution bootstrap sampling.

    Configuration (constructor):
        distribution: ReturnDistribution to draw daily log-returns from.
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
        """Forecast terminal price distribution at horizon_days from spot."""
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")

        draws = self.rng.choice(
            self.distribution.samples,
            size=(self.n_paths, horizon_days),
            replace=True,
            p=self.distribution.weights,
        )
        # Log-return additivity: sum of K iid daily log-returns = K-day log-return.
        k_day_log_returns = draws.sum(axis=1)
        terminal_prices = spot * np.exp(k_day_log_returns)

        logger.debug(
            "BootstrapForecaster: n_paths=%d horizon=%d spot=%.4f -> "
            "mean=%.4f std=%.4f",
            self.n_paths, horizon_days, spot,
            float(terminal_prices.mean()), float(terminal_prices.std()),
        )
        return PriceDistribution(prices=terminal_prices)
