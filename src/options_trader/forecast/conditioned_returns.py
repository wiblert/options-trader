"""
conditioned_returns.py — partition a return series into per-event-type distributions.

The bridge between the event layer and the event-aware forecaster. Given a
StockReturnTS and an EventCalendar, it partitions the daily log-returns into:
    * `normal`  — returns on non-event days (a ReturnDistribution)
    * `by_type` — one ReturnDistribution per event type that has >= 1 historical sample

The forecaster then routes each horizon day's draw via `distribution_for(type)`:
event days draw from the matching type's distribution, normal days from `normal`.

This lives in forecast/ (not data/) because it composes ReturnDistribution — a
forecasting concern. ReturnDistribution itself is NEVER modified: it is simply
constructed from different index subsets of the same log-return array. That keeps it
the pure data container we decided it should remain.

Empty-calendar degradation: with no events, `normal` is the FULL return series and
`by_type` is empty, so an all-None horizon schedule reproduces the plain
BootstrapForecaster (distributionally).

Per-ticker sample policy (V1): each type's distribution uses only this ticker's own
historical occurrences (~12 earnings over 3y) — thin and high-variance by design;
`event_window` can widen each event to a ±N-day window to capture post-event path
shape and multiply samples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.event import EventType
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.return_distribution import (
    DEFAULT_DECAY_LAMBDA,
    ReturnDistribution,
)


logger = logging.getLogger(__name__)


@dataclass
class ConditionedReturnDistributions:
    """A `normal` ReturnDistribution plus one per event type present in history."""

    normal: ReturnDistribution
    by_type: dict[EventType, ReturnDistribution]

    def distribution_for(self, event_type: Optional[EventType]) -> ReturnDistribution:
        """Route to the type-keyed distribution, falling back to `normal`.

        None (a non-event day) -> normal. A type with no historical samples was
        dropped from `by_type`, so it also falls back to normal.
        """
        if event_type is None:
            return self.normal
        return self.by_type.get(event_type, self.normal)

    @classmethod
    def build(
        cls,
        ts: StockReturnTS,
        calendar: EventCalendar,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        event_window: int = 0,
    ) -> "ConditionedReturnDistributions":
        """Partition `ts`'s daily log-returns by event type using `calendar`.

        Args:
            ts: price series to derive log-returns from.
            calendar: events mapped onto the return indices.
            decay_lambda: passed through to each ReturnDistribution.
            event_window: include ±event_window returns around each event index
                (default 0 = the single event-day return only).
        """
        if event_window < 0:
            raise ValueError(f"event_window must be >= 0, got {event_window}")

        close = np.asarray(ts.close, dtype=float)
        log_returns = np.diff(np.log(close))
        n = len(log_returns)
        if n < 1:
            raise ValueError(f"need >= 2 prices to form returns, got {len(close)}")

        idx_by_type = calendar.event_return_indices(ts)

        # Build the union event mask (with optional window widening).
        event_mask = np.zeros(n, dtype=bool)
        widened: dict[EventType, np.ndarray] = {}
        for etype, idxs in idx_by_type.items():
            if event_window > 0:
                expanded = np.unique(np.concatenate([
                    np.clip(idxs + d, 0, n - 1) for d in range(-event_window, event_window + 1)
                ]))
            else:
                expanded = idxs
            widened[etype] = expanded
            event_mask[expanded] = True

        # normal = non-event returns. Guard against an empty normal set (all returns
        # are event returns — only possible on tiny synthetic series): fall back to
        # the full series so `normal` is always constructible.
        normal_returns = log_returns[~event_mask]
        if len(normal_returns) == 0:
            logger.warning("all returns are event returns for %s; normal uses full series",
                           ts.ticker)
            normal_returns = log_returns
        normal = ReturnDistribution(normal_returns, decay_lambda=decay_lambda, label=ts.ticker)

        by_type: dict[EventType, ReturnDistribution] = {}
        for etype, idxs in widened.items():
            samples = log_returns[idxs]
            if len(samples) == 0:
                continue
            by_type[etype] = ReturnDistribution(samples, decay_lambda=decay_lambda, label=ts.ticker)

        logger.debug(
            "ConditionedReturnDistributions(%s): normal n=%d, by_type=%s",
            ts.ticker, len(normal_returns),
            {str(t): len(d) for t, d in by_type.items()},
        )
        return cls(normal=normal, by_type=by_type)
