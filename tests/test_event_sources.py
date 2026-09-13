"""Tests for event sources (parsing only — no network).

YFinanceEarningsSource is tested via the pure _rows_to_events helper on a synthetic
DataFrame. ManualCalendarSource + CompositeEventSource are tested against temp CSVs
and fake sources.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_trader.data.events.composite import CompositeEventSource
from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.events.manual_calendar import ManualCalendarSource
from options_trader.data.events.yfinance_source import _rows_to_events


# ---------- YFinanceEarningsSource._rows_to_events ----------

def _earnings_df():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-04-29 16:00", tz="US/Eastern"),  # AMC
        pd.Timestamp("2026-01-28 08:00", tz="US/Eastern"),  # BMO
        pd.Timestamp("2025-10-29 12:00", tz="US/Eastern"),  # midday -> UNKNOWN
    ], name="Earnings Date")
    return pd.DataFrame(
        {"EPS Estimate": [7.2, 8.18, 6.67],
         "Reported EPS": [np.nan, 8.88, 7.25],
         "Surprise(%)": [np.nan, 8.6, 8.66]},
        index=idx,
    )


def test_rows_to_events_types_and_timing():
    events = _rows_to_events(_earnings_df(), "META")
    assert all(e.event_type == EventType.EARNINGS for e in events)
    assert all(e.ticker == "META" for e in events)
    timings = {e.timestamp.hour: e.timing for e in events}
    assert timings[16] == EventTiming.AMC
    assert timings[8] == EventTiming.BMO
    assert timings[12] == EventTiming.UNKNOWN


def test_rows_to_events_metadata():
    events = _rows_to_events(_earnings_df(), "META")
    bmo = next(e for e in events if e.timestamp.hour == 8)
    assert bmo.metadata["surprise_pct"] == pytest.approx(8.6)
    assert bmo.metadata["reported_eps"] == pytest.approx(8.88)
    assert bmo.metadata["source"] == "yfinance"
    amc = next(e for e in events if e.timestamp.hour == 16)
    assert amc.metadata["reported_eps"] is None   # NaN -> None


# ---------- ManualCalendarSource ----------

def _write_csv(tmp_path, rows: str):
    p = tmp_path / "events.csv"
    p.write_text("ticker,date,event_type,timing,note\n" + rows)
    return p


def test_manual_calendar_ticker_filter(tmp_path):
    csv = _write_csv(tmp_path,
                     "AAPL,2026-03-10,fda,bmo,approval\n"
                     "MSFT,2026-03-11,fda,amc,other\n")
    src = ManualCalendarSource(csv)
    events = src.fetch_events("AAPL", date(2026, 1, 1), date(2027, 1, 1))
    assert len(events) == 1
    assert events[0].event_type == EventType.FDA
    assert events[0].timing == EventTiming.BMO


def test_manual_calendar_wildcard_applies_to_all(tmp_path):
    csv = _write_csv(tmp_path, "*,2026-03-18,macro_fomc,amc,FOMC\n")
    src = ManualCalendarSource(csv)
    events = src.fetch_events("ANYTICK", date(2026, 1, 1), date(2027, 1, 1))
    assert len(events) == 1
    assert events[0].ticker == "ANYTICK"
    assert events[0].event_type == EventType.MACRO_FOMC


def test_manual_calendar_window_filter(tmp_path):
    csv = _write_csv(tmp_path,
                     "*,2026-03-18,macro_fomc,amc,in\n"
                     "*,2030-03-18,macro_fomc,amc,out\n")
    src = ManualCalendarSource(csv)
    events = src.fetch_events("X", date(2026, 1, 1), date(2027, 1, 1))
    assert len(events) == 1
    assert events[0].metadata["note"] == "in"


def test_manual_calendar_missing_file_returns_empty(tmp_path):
    src = ManualCalendarSource(tmp_path / "nope.csv")
    assert src.fetch_events("X", date(2026, 1, 1), date(2027, 1, 1)) == []


# ---------- CompositeEventSource ----------

class _FakeSource:
    def __init__(self, events):
        self._events = events

    def fetch_events(self, ticker, start, end):
        return list(self._events)


def test_composite_merges_and_dedups():
    e1 = Event("X", pd.Timestamp("2026-03-18 16:00", tz="US/Eastern"), EventType.MACRO_FOMC)
    e1_dup = Event("X", pd.Timestamp("2026-03-18 16:30", tz="US/Eastern"), EventType.MACRO_FOMC)
    e2 = Event("X", pd.Timestamp("2026-04-29 16:00", tz="US/Eastern"), EventType.EARNINGS)
    comp = CompositeEventSource(_FakeSource([e1]), _FakeSource([e1_dup, e2]))
    out = comp.fetch_events("X", date(2026, 1, 1), date(2027, 1, 1))
    # e1 and e1_dup share (ticker, date, type) -> deduped; e2 distinct.
    assert len(out) == 2
    assert [e.event_type for e in out] == [EventType.MACRO_FOMC, EventType.EARNINGS]


def test_composite_requires_a_source():
    with pytest.raises(ValueError, match="at least one"):
        CompositeEventSource()
