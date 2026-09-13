"""
event_bootstrap_forecaster.py — event-conditioned Monte Carlo bootstrap.

Like BootstrapForecaster, but each horizon day's draw is routed to a distribution
chosen by that day's event type. A forecast over K days with schedule
[None, None, EARNINGS, None, None] draws days 0,1,3,4 from the `normal`
distribution and day 2 from the earnings-day distribution, then sums the K draws
(log-return additivity) and exponentiates — so a known upcoming earnings day
injects an earnings-shaped move at the right place in the horizon.

Design notes:
    * The shared Forecaster.forecast(horizon_days, spot) signature is UNCHANGED. The
      per-day schedule and the conditioned distributions are bound at construction,
      because the forecaster is rebuilt per-iteration by the backtest factory (which
      knows the calendar position). This keeps the ABC clean.
    * Draws are per-column (one rng.choice per horizon day) rather than one 2-D
      choice, since columns may come from different distributions. On an all-None
      schedule with `normal` = full series this is DISTRIBUTIONALLY equivalent to
      BootstrapForecaster (not RNG-bit-identical — the draw order differs).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from options_trader.data.events.event import EventType
from options_trader.forecast.base import Forecaster
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions
from options_trader.forecast.price_distribution import PriceDistribution


logger = logging.getLogger(__name__)


DEFAULT_N_PATHS = 10_000


class EventConditionedBootstrapForecaster(Forecaster):
    """Bootstrap that routes each horizon day's draw by its event type.

    Configuration (constructor):
        conditioned: ConditionedReturnDistributions (normal + per-type distributions).
        event_schedule: length-`horizon` list; entry k is the EventType on horizon
            day k, or None for a normal day. Its length must equal the horizon_days
            passed to forecast().
        n_paths: Monte Carlo paths per forecast (default 10,000).
        seed: optional RNG seed.
    """

    def __init__(
        self,
        conditioned: ConditionedReturnDistributions,
        event_schedule: list[Optional[EventType]],
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
    ) -> None:
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        self.conditioned = conditioned
        self.event_schedule = event_schedule
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if len(self.event_schedule) != horizon_days:
            raise ValueError(
                f"event_schedule length {len(self.event_schedule)} != horizon_days {horizon_days}"
            )

        # Draw one column per horizon day from that day's routed distribution.
        cols = np.empty((self.n_paths, horizon_days), dtype=float)
        for k, etype in enumerate(self.event_schedule):
            dist = self.conditioned.distribution_for(etype)
            cols[:, k] = self.rng.choice(dist.samples, size=self.n_paths, p=dist.weights)

        k_day_log_returns = cols.sum(axis=1)   # log-return additivity
        terminal_prices = spot * np.exp(k_day_log_returns)

        n_event_days = sum(1 for e in self.event_schedule if e is not None)
        logger.debug(
            "EventConditionedBootstrapForecaster: n_paths=%d horizon=%d event_days=%d "
            "spot=%.4f -> mean=%.4f std=%.4f",
            self.n_paths, horizon_days, n_event_days, spot,
            float(terminal_prices.mean()), float(terminal_prices.std()),
        )
        return PriceDistribution(prices=terminal_prices)
