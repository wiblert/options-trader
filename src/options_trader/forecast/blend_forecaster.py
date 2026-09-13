"""
blend_forecaster.py — combine two (or more) forecasters by a fixed-weight MIXTURE.

THE IDEA (to-do "Ensemble / router", approach (d): path-level mixture)
    Event-bootstrap (historical/empirical) and Option-Implied Beta (forward
    option-implied) draw on DIFFERENT information and win on different names. The
    simplest way to combine them is a probability MIXTURE of their terminal price
    distributions: take a fraction w_i of the total probability mass from each
    forecaster's PriceDistribution and stack them into one distribution.

    A 50/50 blend of forecasters A and B is:
        prices  = [A.prices ; B.prices]
        weights = [0.5 · A.weights ; 0.5 · B.weights]
    Because A.weights and B.weights each sum to 1, the mixture sums to 1 — the
    probability still totals 100%. Half the mass is "A's view", half is "B's view";
    each forecaster's full SHAPE (fat tails, skew) is preserved exactly (this is a
    mixture, NOT an average of means or parameters). The mixture mean is the
    weight-average of the component means.

DESIGN
    Fixed equal weights by design (no learned/adaptive combiner) — with limited
    backtest data, fitting blend weights would over-use the data and risk overfitting
    the combiner. `blend_price_distributions` is the reusable pure core; the
    backtest scores the mixture directly via it (no need to instantiate the
    forecaster), and the live `BlendFactory` wires two factories into one.

STATUS: NOT a production default — gated by the standing log-score decision criterion
(CLAUDE.md). Available via `BlendFactory`.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from options_trader.forecast.base import Forecaster
from options_trader.forecast.price_distribution import PriceDistribution


logger = logging.getLogger(__name__)


def blend_price_distributions(
    dists: Sequence[PriceDistribution],
    weights: Optional[Sequence[float]] = None,
) -> PriceDistribution:
    """Mixture of PriceDistributions: stack prices, scale each by its blend weight.

    Args:
        dists: the component PriceDistributions (each already sums to 1).
        weights: per-distribution blend weights (default equal). Need not sum to 1 —
            they are renormalised; must be non-negative with positive sum.

    Returns:
        A PriceDistribution whose weights sum to 1: weight_i · dist_i.weights stacked.
        The mixture mean equals sum_i(norm_weight_i · dist_i.mean()).
    """
    if len(dists) == 0:
        raise ValueError("need at least one distribution to blend")
    if weights is None:
        w = np.full(len(dists), 1.0 / len(dists))
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(dists),):
            raise ValueError(f"weights length {w.shape} != n dists {len(dists)}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights sum to zero")
        w = w / total

    prices = np.concatenate([d.prices for d in dists])
    weights_out = np.concatenate([wi * d.weights for wi, d in zip(w, dists)])
    return PriceDistribution(prices=prices, weights=weights_out)


class BlendForecaster(Forecaster):
    """Mixture forecaster: blend N sub-forecasters' PriceDistributions by fixed weights.

    Constructor:
        forecasters: the component forecasters (typically two).
        weights: blend weights (default equal — i.e. 50/50 for two). Renormalised.

    forecast(horizon, spot) forecasts each component at the same (horizon, spot) and
    returns their probability mixture (see `blend_price_distributions`).
    """

    def __init__(
        self,
        forecasters: Sequence[Forecaster],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if len(forecasters) == 0:
            raise ValueError("need at least one forecaster to blend")
        self.forecasters = list(forecasters)
        # Validate/normalise weights once up front (reuses the helper's checks).
        if weights is None:
            self.weights = np.full(len(self.forecasters), 1.0 / len(self.forecasters))
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (len(self.forecasters),):
                raise ValueError(f"weights length {w.shape} != n forecasters {len(self.forecasters)}")
            if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
                raise ValueError("weights must be finite, non-negative, with positive sum")
            self.weights = w / w.sum()

    @property
    def event_schedule(self):
        """Expose the first sub-forecaster's event schedule, if any, so diagnostics
        like `event_days_in_horizon` (which duck-types on `event_schedule`) still
        report the horizon's event days through the blend wrapper."""
        for f in self.forecasters:
            sched = getattr(f, "event_schedule", None)
            if sched:
                return sched
        return None

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        dists = [f.forecast(horizon_days, spot) for f in self.forecasters]
        blended = blend_price_distributions(dists, self.weights)
        logger.debug(
            "BlendForecaster: %d components weights=%s horizon=%d spot=%.4f -> mean=%.4f std=%.4f",
            len(self.forecasters), np.round(self.weights, 3), horizon_days, spot,
            blended.mean(), blended.std(),
        )
        return blended
