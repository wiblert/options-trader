"""
composite.py — merge several EventSources into one.

V1 usage:
    source = CompositeEventSource(
        YFinanceEarningsSource(),
        ManualCalendarSource("config/manual_events.csv"),
    )

Concatenates the events from each child source, deduplicates events that share
(ticker, calendar date, event_type) — keeping the first occurrence, so earlier
sources win — and returns them sorted ascending by timestamp. A child source that
raises EventSourceError is skipped with a warning rather than failing the whole
fetch (one flaky backend shouldn't blind the others).
"""

from __future__ import annotations

import logging
from datetime import date

from options_trader.data.events.event import Event
from options_trader.data.events.source import EventSource, EventSourceError


logger = logging.getLogger(__name__)


class CompositeEventSource:
    """EventSource that merges + dedups several backends."""

    def __init__(self, *sources: EventSource) -> None:
        if not sources:
            raise ValueError("CompositeEventSource requires at least one source")
        self.sources = sources

    def fetch_events(self, ticker: str, start: date, end: date) -> list[Event]:
        merged: list[Event] = []
        seen: set[tuple[str, date, str]] = set()
        for src in self.sources:
            try:
                events = src.fetch_events(ticker, start, end)
            except EventSourceError as exc:
                logger.warning("source %s failed for %s, skipping: %s",
                               type(src).__name__, ticker, exc)
                continue
            for e in events:
                key = (e.ticker.upper(),
                       e.timestamp.tz_convert("US/Eastern").date(),
                       str(e.event_type))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(e)
        merged.sort(key=lambda e: e.timestamp)
        return merged
