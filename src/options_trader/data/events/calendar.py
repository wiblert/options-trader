"""
calendar.py — EventCalendar: map events onto a return series + build horizon schedules.

This is the load-bearing geometry of event conditioning. Two questions it answers:

1. HISTORICAL — "which daily-return indices were event returns?" Used to partition
   history into per-type return distributions (conditioned_returns.py).
2. FORWARD — "for a forecast made at close of day D over the next K trading days,
   which horizon day carries which event type?" Used to route bootstrap draws.

Return-index convention (must match log_score.py / numpy diff):
    log_returns = np.diff(np.log(close))  -> length n-1
    return index i is the move close[i] -> close[i+1], i.e. "the return OF trading
    day dates[i+1]". So the return of the trading day at position p is index p-1.

Event -> return-index attribution (uses Event.timing):
    Let q = searchsorted(dates, event_date). If the event date is a trading day
    (dates[q] == event_date):
        AMC / UNKNOWN  -> index q     (move realised on the NEXT session's return)
        BMO            -> index q-1   (move realised on the SAME day's return)
    If the event date is a holiday/weekend (not in dates):
        -> index q-1   (absorbed into the next available session's return)
    Indices outside [0, n-2] (event at/before the first return or after the last)
    are dropped.

UNKNOWN timing is treated as AMC (the large-cap earnings norm).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.events.source import EventSource
from options_trader.data.stock_return_ts import StockReturnTS


logger = logging.getLogger(__name__)


# Precedence when multiple event types land on the same return index (highest first).
# One label per horizon day; the winner is the earliest in this order.
TYPE_PRECEDENCE: tuple[EventType, ...] = (
    EventType.EARNINGS,
    EventType.FDA,
    EventType.MACRO_FOMC,
    EventType.MACRO_CPI,
    EventType.DIVIDEND_EX,
)


class EventCalendar:
    """Per-ticker collection of Events with return-series mapping helpers."""

    def __init__(self, ticker: str, events: list[Event]) -> None:
        for e in events:
            if e.ticker.upper() != ticker.upper():
                raise ValueError(
                    f"EventCalendar({ticker}) got event for {e.ticker}"
                )
        self.ticker = ticker
        self.events = sorted(events, key=lambda e: e.timestamp)

    @classmethod
    def from_source(
        cls, source: EventSource, ticker: str, start: date, end: date
    ) -> "EventCalendar":
        return cls(ticker, source.fetch_events(ticker, start, end))

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"EventCalendar({self.ticker}, {len(self.events)} events)"

    # ------------------------------------------------------------------ #
    # internal: one event -> return index (or None if out of range)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _return_index_for_event(
        event: Event, dates: np.ndarray, n_returns: int
    ) -> Optional[int]:
        # Normalise event datetime to a tz-naive US/Eastern midnight, matching how
        # history.py stores StockReturnTS.dates.
        d = event.timestamp.tz_convert("US/Eastern").normalize().tz_localize(None)
        d64 = pd.Timestamp(d).to_datetime64()
        q = int(np.searchsorted(dates, d64, side="left"))
        is_trading_day = q < len(dates) and dates[q] == d64
        amc = event.timing in (EventTiming.AMC, EventTiming.UNKNOWN)
        if is_trading_day:
            idx = q if amc else q - 1
        else:
            idx = q - 1
        if idx < 0 or idx >= n_returns:
            return None
        return idx

    # ------------------------------------------------------------------ #
    # historical mapping
    # ------------------------------------------------------------------ #
    def event_return_indices(
        self, ts: StockReturnTS, event_type: Optional[EventType] = None
    ) -> dict[EventType, np.ndarray]:
        """Map events onto the indices of the daily returns they moved.

        Returns {EventType: sorted unique int index array} for each type that has
        at least one in-range event. If `event_type` is given, only that type.
        """
        dates = ts.dates
        n_returns = len(dates) - 1
        if n_returns < 1:
            return {}

        out: dict[EventType, list[int]] = {}
        n_unknown = 0
        for e in self.events:
            if event_type is not None and e.event_type != event_type:
                continue
            if e.timing == EventTiming.UNKNOWN:
                n_unknown += 1
            idx = self._return_index_for_event(e, dates, n_returns)
            if idx is None:
                continue
            out.setdefault(e.event_type, []).append(idx)
        if n_unknown:
            logger.warning(
                "%d event(s) for %s have UNKNOWN timing; treated as AMC (next-day return)",
                n_unknown, self.ticker,
            )
        return {t: np.unique(np.asarray(v, dtype=int)) for t, v in out.items()}

    # ------------------------------------------------------------------ #
    # forward mapping
    # ------------------------------------------------------------------ #
    def horizon_event_schedule(
        self,
        ts: StockReturnTS,
        start_idx: int,
        horizon_days: int,
        future_dates: Optional[np.ndarray] = None,
    ) -> list[Optional[EventType]]:
        """Per-horizon-day event labels for a forecast made at close of dates[start_idx].

        The horizon spans return indices [start_idx, start_idx + horizon_days). Entry
        k of the returned list is the EventType landing on return index start_idx+k
        (resolved by TYPE_PRECEDENCE on collision), or None.

        For TRUE forward forecasting (a live run, horizon beyond loaded history) use
        `forward_schedule(run_date, horizon_days)` instead — there the horizon days
        do not exist in `ts` yet and must be projected from the calendar.
        """
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if future_dates is not None:
            raise NotImplementedError(
                "in-history horizon_event_schedule does not take future_dates; "
                "use forward_schedule(run_date, horizon_days) for live forecasting"
            )
        n_returns = len(ts.dates) - 1
        if start_idx < 0 or start_idx + horizon_days > n_returns:
            raise ValueError(
                f"horizon [{start_idx}, {start_idx + horizon_days}) is out of range "
                f"for {n_returns} returns"
            )

        idx_by_type = self.event_return_indices(ts)
        # Invert to {return_index: [types]} for the horizon window only.
        schedule: list[Optional[EventType]] = [None] * horizon_days
        for k in range(horizon_days):
            ri = start_idx + k
            claimants = [t for t, idxs in idx_by_type.items() if ri in idxs]
            schedule[k] = self._resolve_precedence(claimants)
        return schedule

    def forward_schedule(
        self, run_date: date, horizon_days: int
    ) -> list[Optional[EventType]]:
        """Per-horizon-day event labels for a LIVE forecast made on `run_date`.

        Unlike `horizon_event_schedule` (which indexes into already-loaded history),
        this projects the next `horizon_days` trading days — which do not exist in any
        StockReturnTS yet — and maps the calendar's KNOWN future events (e.g. an
        announced earnings date) onto them. Legitimate: earnings dates are published
        in advance, so using them is not look-ahead.

        Horizon day k is the return of the (k+1)-th trading day after `run_date`
        (day 0 = the move from run_date's close to the next session). The same
        BMO/AMC attribution as the historical path is reused by laying the events on
        a synthetic date axis [run_date, F0, F1, ...] and calling
        `_return_index_for_event` — the return-index convention makes the resulting
        index identical to the horizon day. Entry k is the EventType on day k (by
        TYPE_PRECEDENCE on collision) or None.
        """
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")

        rd64 = np.datetime64(run_date, "D")
        # First trading day STRICTLY after run_date (or on/after it if run_date is
        # itself a non-trading day, e.g. a weekend run).
        b0 = np.busday_offset(rd64, 0, roll="forward")
        first = 1 if b0 == rd64 else 0
        future = np.busday_offset(b0, np.arange(first, first + horizon_days))
        # Synthetic axis: [run_date, F0, ..., F_{K-1}] -> K returns whose index == horizon day.
        combined = np.concatenate([np.array([rd64]), future]).astype("datetime64[ns]")

        claimants: dict[int, list[EventType]] = {}
        for e in self.events:
            idx = self._return_index_for_event(e, combined, horizon_days)
            if idx is None:
                continue
            claimants.setdefault(idx, []).append(e.event_type)
        return [self._resolve_precedence(claimants.get(k, [])) for k in range(horizon_days)]

    @staticmethod
    def _resolve_precedence(claimants: list[EventType]) -> Optional[EventType]:
        if not claimants:
            return None
        for t in TYPE_PRECEDENCE:
            if t in claimants:
                return t
        return claimants[0]  # any type not in the precedence list: first wins
