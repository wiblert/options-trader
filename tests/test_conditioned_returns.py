"""Tests for ConditionedReturnDistributions (the event->distribution bridge)."""

from datetime import date

import numpy as np
import pandas as pd

from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions


DATES = pd.bdate_range("2026-01-05", periods=12)


def _ts(close=None) -> StockReturnTS:
    n = len(DATES)
    if close is None:
        close = np.linspace(100.0, 112.0, n)
    close = np.asarray(close, dtype=float)
    return StockReturnTS(
        ticker="TEST", dates=DATES.to_numpy(),
        open=close, high=close, low=close, close=close, volume=np.ones(n),
        source="synthetic", start=date(2026, 1, 5), end=date(2026, 1, 21),
    )


def _earnings(day: str) -> Event:
    return Event("TEST", pd.Timestamp(f"{day} 16:00", tz="US/Eastern"),
                 EventType.EARNINGS, EventTiming.AMC)


def test_empty_calendar_normal_is_full_series():
    ts = _ts()
    cond = ConditionedReturnDistributions.build(ts, EventCalendar("TEST", []))
    n_returns = len(ts.close) - 1
    assert len(cond.normal) == n_returns
    assert cond.by_type == {}


def test_partition_is_disjoint_and_covers_all():
    ts = _ts()
    cal = EventCalendar("TEST", [_earnings("2026-01-08"), _earnings("2026-01-14")])
    cond = ConditionedReturnDistributions.build(ts, cal)
    n_returns = len(ts.close) - 1
    n_event = len(cond.by_type[EventType.EARNINGS])
    assert n_event == 2
    assert len(cond.normal) + n_event == n_returns   # disjoint, complete


def test_distribution_for_routing():
    ts = _ts()
    cal = EventCalendar("TEST", [_earnings("2026-01-08")])
    cond = ConditionedReturnDistributions.build(ts, cal)
    assert cond.distribution_for(EventType.EARNINGS) is cond.by_type[EventType.EARNINGS]
    assert cond.distribution_for(None) is cond.normal
    # A type absent from history falls back to normal.
    assert cond.distribution_for(EventType.FDA) is cond.normal


def test_event_window_widens_sample_count():
    ts = _ts()
    cal = EventCalendar("TEST", [_earnings("2026-01-09")])  # one interior event
    narrow = ConditionedReturnDistributions.build(ts, cal, event_window=0)
    wide = ConditionedReturnDistributions.build(ts, cal, event_window=1)
    assert len(narrow.by_type[EventType.EARNINGS]) == 1
    assert len(wide.by_type[EventType.EARNINGS]) == 3   # idx-1, idx, idx+1


def test_earnings_samples_are_the_right_returns():
    # Build close so the earnings-day return is distinct and checkable.
    close = np.array([100, 101, 102, 130, 104, 105, 106, 107, 108, 109, 110, 111], dtype=float)
    ts = _ts(close)
    log_returns = np.diff(np.log(close))
    # AMC on dates[2] (2026-01-07) -> return index 2 = the move close[2]->close[3] (the +130 jump).
    cal = EventCalendar("TEST", [_earnings("2026-01-07")])
    cond = ConditionedReturnDistributions.build(ts, cal)
    np.testing.assert_allclose(cond.by_type[EventType.EARNINGS].samples, [log_returns[2]])
