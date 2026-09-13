"""Tests for FomcCalendarSource — the auto-updating Fed-meeting EventSource.

Network is stubbed: the pure parser runs on a saved HTML sample, and fetch_events
is driven through a fake _get() so no request hits the Fed.
"""

from datetime import date

import pytest

from options_trader.data.events.event import EventTiming, EventType
from options_trader.data.events.source import EventSourceError
from options_trader.data.events.fomc_source import FomcCalendarSource, _parse_fomc_calendar


# Trimmed sample mirroring the real Fed page: two year panels, a cross-month meeting
# ("Apr/May" + "30-1" -> May 1), a SEP-marker meeting ("17-18*"), and an unscheduled
# "notation vote" row that must be excluded.
_SAMPLE_HTML = """
<div class="panel panel-default"><div class="panel-heading"><h4><a id="111">2025 FOMC Meetings</a></h4></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month col-md-2"><strong>January</strong></div>
    <div class="fomc-meeting__date col-md-10">28-29</div></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month col-md-2"><strong>Apr/May</strong></div>
    <div class="fomc-meeting__date col-md-10">30-1</div></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month col-md-2"><strong>August</strong></div>
    <div class="fomc-meeting__date col-md-10">22 (notation vote)</div></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month col-md-2"><strong>September</strong></div>
    <div class="fomc-meeting__date col-md-10">16-17*</div></div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="222">2026 FOMC Meetings</a></h4></div>
  <div class="row fomc-meeting">
    <div class="fomc-meeting__month col-md-2"><strong>June</strong></div>
    <div class="fomc-meeting__date col-md-10">16-17*</div></div>
</div>
"""


# ---------- pure parser ----------

def test_parse_extracts_announcement_last_day():
    dates = _parse_fomc_calendar(_SAMPLE_HTML)
    assert date(2025, 1, 29) in dates        # 28-29 -> 29
    assert date(2025, 9, 17) in dates        # 16-17* -> 17 (marker stripped)
    assert date(2026, 6, 17) in dates        # second-year panel parsed


def test_parse_cross_month_meeting():
    dates = _parse_fomc_calendar(_SAMPLE_HTML)
    assert date(2025, 5, 1) in dates         # "Apr/May" + "30-1" -> May 1


def test_parse_excludes_notation_vote():
    dates = _parse_fomc_calendar(_SAMPLE_HTML)
    assert not any(d.month == 8 for d in dates)


def test_parse_sorted_and_unique():
    dates = _parse_fomc_calendar(_SAMPLE_HTML)
    assert dates == sorted(set(dates))


def test_parse_empty_html_returns_empty():
    assert _parse_fomc_calendar("<html>no meetings here</html>") == []


# ---------- fetch_events (stubbed network) ----------

def _stub(monkeypatch, html=_SAMPLE_HTML):
    monkeypatch.setattr(FomcCalendarSource, "_get", lambda self: html)


def test_fetch_filters_window_and_tags_ticker(monkeypatch):
    _stub(monkeypatch)
    src = FomcCalendarSource()
    events = src.fetch_events("NVDA", date(2025, 1, 1), date(2025, 12, 31))
    # 2026-06 is outside the window; the notation vote is excluded.
    dts = sorted(e.timestamp.tz_convert("US/Eastern").date() for e in events)
    assert dts == [date(2025, 1, 29), date(2025, 5, 1), date(2025, 9, 17)]
    assert all(e.ticker == "NVDA" for e in events)          # macro tagged to the ticker
    assert all(e.event_type == EventType.MACRO_FOMC for e in events)


def test_fetch_window_is_half_open(monkeypatch):
    _stub(monkeypatch)
    src = FomcCalendarSource()
    # end is exclusive: a window ending exactly on a meeting date drops it.
    events = src.fetch_events("AAPL", date(2025, 1, 1), date(2025, 1, 29))
    assert all(e.timestamp.tz_convert("US/Eastern").date() != date(2025, 1, 29) for e in events)


def test_fetch_default_timing_is_announcement_day(monkeypatch):
    _stub(monkeypatch)
    src = FomcCalendarSource()
    events = src.fetch_events("AAPL", date(2025, 1, 1), date(2026, 12, 31))
    # BMO mapping = announcement day's own close-to-close return (see fomc_source docstring).
    assert all(e.timing == EventTiming.BMO for e in events)


def test_fetch_raises_on_network_error(monkeypatch):
    def boom(self):
        raise ConnectionError("dns fail")
    monkeypatch.setattr(FomcCalendarSource, "_get", boom)
    with pytest.raises(EventSourceError, match="FOMC calendar fetch failed"):
        FomcCalendarSource().fetch_events("AAPL", date(2025, 1, 1), date(2026, 1, 1))


def test_fetch_raises_when_parse_finds_nothing(monkeypatch):
    _stub(monkeypatch, html="<html>structure changed</html>")
    with pytest.raises(EventSourceError, match="parsed 0 meetings"):
        FomcCalendarSource().fetch_events("AAPL", date(2025, 1, 1), date(2026, 1, 1))
