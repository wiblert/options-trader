"""
gaussian_forecaster.py — Black-Scholes-equivalent baseline forecaster.

Assumes daily log-returns are iid N(μ, σ²). μ and σ² are estimated from the
weighted sample mean and variance of the supplied ReturnDistribution. The
K-day log-return is N(K·μ, K·σ²) by iid additivity, and the terminal price
is lognormal:

    R_K ~ N(K·μ, K·σ²)
    S_T = spot · exp(R_K)

This is the standard "constant-vol Gaussian iid" assumption that Black-Scholes
relies on. Comparing log scores against the BootstrapForecaster quantifies the
value of empirical-distribution shape (fat tails, skew) over the Gaussian
assumption.

Note on weighting:
    The forecaster uses the *weighted* mean and variance of the distribution it
    is given. Pass a uniform-weight ReturnDistribution for the canonical BS
    estimate; pass a decay-weighted one for a RiskMetrics-style estimate. In
    a head-to-head against BootstrapForecaster, give both the SAME distribution
    so the only difference is the parametric assumption.
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
MIN_VARIANCE = 1e-12   # sigma below ~1e-6 daily log-return is degenerate


class GaussianForecaster(Forecaster):
    """Black-Scholes-equivalent: iid Gaussian log-returns with μ, σ² from
    the weighted sample moments of the supplied ReturnDistribution."""

    def __init__(
        self,
        distribution: ReturnDistribution,
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
    ) -> None:
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")

        # Weighted mean and variance of the daily log-returns.
        mu = float(np.sum(distribution.samples * distribution.weights))
        var = float(np.sum(distribution.weights * (distribution.samples - mu) ** 2))
        if var < MIN_VARIANCE:
            raise ValueError(
                f"weighted variance of distribution is degenerate "
                f"(var={var:.3e} < {MIN_VARIANCE:.0e}); GaussianForecaster needs "
                f"positive variance to produce a meaningful forecast"
            )

        self.mu = mu
        self.sigma = float(np.sqrt(var))
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")

        # K-day log-return ~ N(K·μ, K·σ²) by iid additivity
        k_mean = self.mu * horizon_days
        k_std = self.sigma * np.sqrt(horizon_days)
        log_returns = self.rng.normal(loc=k_mean, scale=k_std, size=self.n_paths)
        terminal_prices = spot * np.exp(log_returns)

        logger.debug(
            "GaussianForecaster: μ=%.5f σ=%.5f horizon=%d spot=%.4f "
            "→ k_mean=%.5f k_std=%.5f",
            self.mu, self.sigma, horizon_days, spot, k_mean, k_std,
        )
        return PriceDistribution(prices=terminal_prices)
