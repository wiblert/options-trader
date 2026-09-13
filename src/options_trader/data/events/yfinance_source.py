"""
yfinance_source.py — earnings events from yfinance.

yfinance's get_earnings_dates() returns a DataFrame indexed by the tz-aware earnings
datetime with columns 'EPS Estimate', 'Reported EPS', 'Surprise(%)', covering BOTH
historical and upcoming announcements — exactly what event conditioning needs.

The row->Event parsing is isolated in the pure `_rows_to_events()` helper so it can
be unit-tested on a synthetic DataFrame without any network call (mirrors how
test_history.py avoids live calls).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.events.source import EventSourceError


logger = logging.getLogger(__name__)


DEFAULT_LIMIT = 40   # ~10 years of quarterly earnings; plenty of history + upcoming
AMC_HOUR = 16        # >= 16:00 ET -> after market close
BMO_HOUR = 10        # <= 10:00 ET -> before/at market open


def _timing_from_timestamp(ts: pd.Timestamp) -> EventTiming:
    """Infer BMO/AMC from the announcement time of day (ET)."""
    hour = ts.hour
    if hour >= AMC_HOUR:
        return EventTiming.AMC
    if hour <= BMO_HOUR:
        return EventTiming.BMO
    return EventTiming.UNKNOWN


def _opt_float(value) -> float | None:
    """Return a finite float or None (NaN/None -> None)."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _rows_to_events(df: pd.DataFrame, ticker: str) -> list[Event]:
    """Convert a get_earnings_dates() DataFrame into Events (pure, testable).

    The DataFrame is indexed by tz-aware earnings datetime. Rows whose index is
    tz-naive are localised to US/Eastern (yfinance normally returns tz-aware).
    """
    events: list[Event] = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        if ts.tzinfo is None:
            ts = ts.tz_localize("US/Eastern")
        metadata = {
            "source": "yfinance",
            "eps_estimate": _opt_float(row.get("EPS Estimate")),
            "reported_eps": _opt_float(row.get("Reported EPS")),
            "surprise_pct": _opt_float(row.get("Surprise(%)")),
        }
        events.append(Event(
            ticker=ticker,
            timestamp=ts,
            event_type=EventType.EARNINGS,
            timing=_timing_from_timestamp(ts),
            metadata=metadata,
        ))
    return events


class YFinanceEarningsSource:
    """EventSource backend: earnings dates via yfinance (historical + upcoming)."""

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self.limit = limit

    def fetch_events(self, ticker: str, start: date, end: date) -> list[Event]:
        import yfinance as yf  # local import: keep module import cheap / offline-testable

        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=self.limit)
        except Exception as exc:  # noqa: BLE001 — yfinance raises a zoo of types
            raise EventSourceError(f"yfinance earnings fetch failed for {ticker}: {exc!r}") from exc
        if df is None or df.empty:
            logger.warning("yfinance returned no earnings dates for %s", ticker)
            return []

        events = _rows_to_events(df, ticker)
        # Window filter on calendar date; [start, end).
        events = [e for e in events if start <= e.timestamp.tz_convert("US/Eastern").date() < end]
        events.sort(key=lambda e: e.timestamp)
        logger.info("YFinanceEarningsSource: %d earnings events for %s in [%s, %s)",
                    len(events), ticker, start, end)
        return events
