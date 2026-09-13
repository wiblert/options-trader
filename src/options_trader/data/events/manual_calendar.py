"""
manual_calendar.py — events from a hand-curated CSV.

Covers catalysts with no clean structured API (macro prints, FDA decisions). The CSV
is committed and offline, so it's deterministic and unit-testable without network.

CSV schema (header required):
    ticker,date,event_type,timing,note
    AAPL,2026-05-13,macro_cpi,bmo,May CPI print
    *,2026-06-18,macro_fomc,amc,FOMC decision

Rows:
    ticker      — symbol, or '*' to apply to every ticker (broad macro events).
    date        — YYYY-MM-DD (interpreted in US/Eastern).
    event_type  — an EventType value (e.g. macro_cpi, fda).
    timing      — bmo / amc / unknown.
    note        — free text -> Event.metadata['note'].
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.events.source import EventSourceError


logger = logging.getLogger(__name__)


WILDCARD_TICKER = "*"
REQUIRED_COLUMNS = {"ticker", "date", "event_type"}


class ManualCalendarSource:
    """EventSource backend reading events from a committed CSV file."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def fetch_events(self, ticker: str, start: date, end: date) -> list[Event]:
        if not self.csv_path.exists():
            logger.warning("manual events CSV not found at %s; returning no events", self.csv_path)
            return []
        try:
            df = pd.read_csv(self.csv_path, dtype=str).fillna("")
        except Exception as exc:  # noqa: BLE001
            raise EventSourceError(f"failed to read manual events CSV {self.csv_path}: {exc!r}") from exc

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise EventSourceError(f"manual events CSV {self.csv_path} missing columns: {missing}")

        events: list[Event] = []
        for _, row in df.iterrows():
            row_ticker = row["ticker"].strip()
            if row_ticker != WILDCARD_TICKER and row_ticker.upper() != ticker.upper():
                continue
            d = pd.Timestamp(row["date"].strip())
            if d.tzinfo is None:
                d = d.tz_localize("US/Eastern")
            if not (start <= d.tz_convert("US/Eastern").date() < end):
                continue
            timing_raw = (row.get("timing") or "").strip() or EventTiming.UNKNOWN.value
            events.append(Event(
                ticker=ticker,
                timestamp=d,
                event_type=EventType(row["event_type"].strip()),
                timing=EventTiming(timing_raw),
                metadata={"source": "manual", "note": (row.get("note") or "").strip()},
            ))
        events.sort(key=lambda e: e.timestamp)
        logger.info("ManualCalendarSource: %d events for %s in [%s, %s)",
                    len(events), ticker, start, end)
        return events
