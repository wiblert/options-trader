"""
drift_dampened_bootstrap_forecaster.py — bootstrap forecast with the terminal
mean shrunk toward spot (a drift-dampening experiment).

MOTIVATION (see docs/issues.md ISSUE-1)
    The plain bootstrap inherits the trailing recency-weighted historical drift,
    so its terminal distribution is centred at `spot · E[exp(ΣR)]`, which sits
    above/below spot purely from past momentum. The trailing sample mean is a
    notoriously high-variance estimate of true drift, so that centre is fragile.
    This forecaster lets us shrink the centre back toward the martingale value
    (today's spot) by a tunable fraction and measure — via the rolling log-score
    backtest — whether removing some of the drift improves out-of-sample
    calibration.

MECHANISM (deliberately a pure LINEAR TRANSLATION, per the experiment spec)
    1. Draw the terminal prices exactly as BootstrapForecaster does.
    2. Let m = mean of those terminal prices.
    3. Shift EVERY terminal price by the same amount:

           shift = drift_dampening · (spot − m)
           price'_i = price_i + shift

       so the new mean is (1 − κ)·m + κ·spot, with κ = drift_dampening:
         κ = 0  → no change (identical to BootstrapForecaster)
         κ = 1  → mean moved all the way to spot (zero net drift)
         0<κ<1  → partial shrinkage.

    Because it is an additive translation, the DISPERSION and SHAPE (variance,
    skew, fat tails) of the bootstrap are preserved exactly — only the location
    moves. That isolates the effect of drift from the effect of shape, which is
    the whole point of the experiment.

    Contrast with a log-space drift shrink (scaling the per-day μ): that would be
    multiplicative and would also rescale dispersion. We chose the additive
    translation on purpose; a log-space variant is a separate experiment.

CAVEAT
    A large negative shift (dampening a big positive drift) can in principle push
    the lowest terminal prices to/through zero. At equity horizons the shift
    (κ·|mean−spot|, a few tenths of a percent of spot over a 5-day horizon) is far
    smaller than the gap to the lowest path, so this effectively never fires; we
    raise a clear error if it ever does rather than silently clipping.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from options_trader.forecast.bootstrap_forecaster import (
    DEFAULT_N_PATHS,
    BootstrapForecaster,
)
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


logger = logging.getLogger(__name__)


class DriftDampenedBootstrapForecaster(BootstrapForecaster):
    """BootstrapForecaster whose terminal mean is shrunk toward spot by a
    `drift_dampening` fraction κ ∈ [0, 1] via a pure additive translation."""

    def __init__(
        self,
        distribution: ReturnDistribution,
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
        drift_dampening: float = 0.0,
    ) -> None:
        if not (0.0 <= drift_dampening <= 1.0):
            raise ValueError(
                f"drift_dampening must be in [0, 1], got {drift_dampening}"
            )
        super().__init__(distribution, n_paths=n_paths, seed=seed)
        self.drift_dampening = float(drift_dampening)

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        # Same draw as the plain bootstrap (CRN-compatible: identical RNG usage).
        base = super().forecast(horizon_days, spot)
        kappa = self.drift_dampening
        if kappa == 0.0:
            return base

        m = base.mean()                       # uniform-weight mean of the draws
        shift = kappa * (spot - m)
        shifted = base.prices + shift

        if np.any(shifted <= 0):
            n_bad = int((shifted <= 0).sum())
            raise ValueError(
                f"drift-dampening shift {shift:+.4f} (κ={kappa}, spot={spot:.4f}, "
                f"mean={m:.4f}) drove {n_bad} terminal price(s) <= 0; "
                f"translation too large to keep prices positive"
            )

        logger.debug(
            "DriftDampened: κ=%.3f spot=%.4f mean=%.4f shift=%+.4f -> new_mean=%.4f",
            kappa, spot, m, shift, m + shift,
        )
        # weights are uniform from the bootstrap; preserve them under translation.
        return PriceDistribution(prices=shifted, weights=base.weights)
