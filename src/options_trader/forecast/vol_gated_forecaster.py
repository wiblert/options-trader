"""
vol_gated_forecaster.py — use GARCH-FHS only when conditional vol diverges from
baseline; fall back to the plain bootstrap otherwise.

WHY
    The 40-name backtest (Session 11) showed GARCH-FHS is a CONVEX win: it helps
    dramatically on names that hit a vol-regime change / fat-tail event, but adds
    a slight drag on calm, steady names (where today's conditional vol ≈ the
    unconditional vol the bootstrap already assumes, so the extra machinery is
    pure noise). The gate keeps the convex wins and removes the calm-name drag by
    routing each ticker-day to the forecaster that should help it.

GATE
    g = σ_{T+1} / σ̄   (conditional one-step-ahead vol ÷ GARCH long-run vol)
      use FHS        if g ≥ vol_threshold τ   (vol has diverged — bootstrap stale)
      use bootstrap  otherwise                (calm regime — bootstrap's fixed
                                               width is fine, skip the noise)

    τ is a hyperparameter fit by the log-score backtest (sweep via
    `run_vol_gate_experiment`). The endpoints recover the two pure forecasters:
        τ = 0.0  → g ≥ 0 always → ALWAYS FHS        (== GarchFhsForecaster)
        τ = ∞    → g ≥ ∞ never  → ALWAYS bootstrap  (== BootstrapForecaster, incumbent)
    so the sweep interpolates between them and finds the crossover.

    The decision is made once at construction (per ticker-day) from the GARCH fit;
    `forecast()` just delegates to the chosen sub-forecaster. Both sub-forecasters
    are built with the SAME seed, so a gated arm that picks bootstrap sees the
    identical RNG stream as the pure-bootstrap baseline (common random numbers),
    and a gated arm that picks FHS matches the pure-FHS arm — the paired log-score
    difference isolates the gating decision.
"""

from __future__ import annotations

from typing import Optional

from options_trader.forecast.base import Forecaster
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.garch_fhs_forecaster import (
    DEFAULT_N_PATHS,
    GarchFhsForecaster,
)
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


class VolGatedForecaster(Forecaster):
    """Route to GARCH-FHS when σ_{T+1}/σ̄ ≥ vol_threshold, else the plain bootstrap."""

    def __init__(
        self,
        distribution: ReturnDistribution,
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
        vol_threshold: float = 1.0,
    ) -> None:
        if vol_threshold < 0:
            raise ValueError(f"vol_threshold must be >= 0, got {vol_threshold}")
        self.vol_threshold = float(vol_threshold)
        # Same seed for both => CRN with the pure arms in the sweep.
        self._fhs = GarchFhsForecaster(distribution, n_paths=n_paths, seed=seed)
        self._bootstrap = BootstrapForecaster(distribution, n_paths=n_paths, seed=seed)
        self.vol_ratio = self._fhs.vol_ratio
        self.use_fhs = self.vol_ratio >= self.vol_threshold

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        chosen = self._fhs if self.use_fhs else self._bootstrap
        return chosen.forecast(horizon_days, spot)
