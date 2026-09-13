"""Tests for EventCalendar return-index mapping and horizon scheduling.

Synthetic StockReturnTS over 10 consecutive business days (no network). Index
convention: return index i is the move dates[i] -> dates[i+1].
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.stock_return_ts import StockReturnTS


# 10 business days: 2026-01-05 (Mon) .. 2026-01-16 (Fri). Sat 01-10/Sun 01-11 are gaps.
DATES = pd.bdate_range("2026-01-05", periods=10)


def _ts() -> StockReturnTS:
    n = len(DATES)
    close = np.linspace(100.0, 110.0, n)
    return StockReturnTS(
        ticker="TEST",
        dates=DATES.to_numpy(),
        open=close, high=close, low=close, close=close, volume=np.ones(n),
        source="synthetic", start=date(2026, 1, 5), end=date(2026, 1, 17),
    )


def _event(day: str, timing: EventTiming, etype=EventType.EARNINGS, hour=16) -> Event:
    return Event("TEST", pd.Timestamp(f"{day} {hour:02d}:00", tz="US/Eastern"), etype, timing)


def test_amc_event_maps_to_next_day_return():
    # AMC on dates[3] (2026-01-08) -> move on next session's return -> index 3.
    cal = EventCalendar("TEST", [_event("2026-01-08", EventTiming.AMC)])
    idx = cal.event_return_indices(_ts())
    np.testing.assert_array_equal(idx[EventType.EARNINGS], [3])


def test_bmo_event_maps_to_same_day_return():
    # BMO on dates[3] (2026-01-08) -> move on that day's return -> index 2.
    cal = EventCalendar("TEST", [_event("2026-01-08", EventTiming.BMO, hour=8)])
    idx = cal.event_return_indices(_ts())
    np.testing.assert_array_equal(idx[EventType.EARNINGS], [2])


def test_unknown_timing_treated_as_amc():
    cal = EventCalendar("TEST", [_event("2026-01-08", EventTiming.UNKNOWN, hour=12)])
    idx = cal.event_return_indices(_ts())
    np.testing.assert_array_equal(idx[EventType.EARNINGS], [3])  # same as AMC


def test_holiday_event_maps_to_next_session_return():
    # Event on Sat 2026-01-10 (not a trading day). Next session is dates[5]=01-12.
    # -> next session's return index = 5 - 1 = 4.
    cal = EventCalendar("TEST", [_event("2026-01-10", EventTiming.AMC)])
    idx = cal.event_return_indices(_ts())
    np.testing.assert_array_equal(idx[EventType.EARNINGS], [4])


def test_amc_on_last_day_is_dropped():
    # AMC on dates[9] (last) -> index 9, but max return index is 8 -> dropped.
    cal = EventCalendar("TEST", [_event("2026-01-16", EventTiming.AMC)])
    assert cal.event_return_indices(_ts()) == {}


def test_bmo_on_first_day_is_dropped():
    cal = EventCalendar("TEST", [_event("2026-01-05", EventTiming.BMO, hour=8)])
    assert cal.event_return_indices(_ts()) == {}


def test_horizon_schedule_labels_correct_day():
    cal = EventCalendar("TEST", [_event("2026-01-08", EventTiming.AMC)])  # index 3
    # start_idx=1, horizon=5 -> return indices [1,2,3,4,5]; index 3 is position 2.
    sched = cal.horizon_event_schedule(_ts(), start_idx=1, horizon_days=5)
    assert sched == [None, None, EventType.EARNINGS, None, None]


def test_empty_calendar_schedule_all_none():
    cal = EventCalendar("TEST", [])
    sched = cal.horizon_event_schedule(_ts(), start_idx=1, horizon_days=5)
    assert sched == [None] * 5


def test_collision_precedence_earnings_wins():
    # An earnings (AMC, idx 3) and a macro FOMC (AMC, idx 3) on the same return.
    cal = EventCalendar("TEST", [
        _event("2026-01-08", EventTiming.AMC, EventType.EARNINGS),
        _event("2026-01-08", EventTiming.AMC, EventType.MACRO_FOMC),
    ])
    sched = cal.horizon_event_schedule(_ts(), start_idx=1, horizon_days=5)
    assert sched[2] == EventType.EARNINGS


def test_horizon_out_of_range_raises():
    cal = EventCalendar("TEST", [])
    with pytest.raises(ValueError, match="out of range"):
        cal.horizon_event_schedule(_ts(), start_idx=7, horizon_days=5)  # 7+5 > 9 returns


def test_wrong_ticker_event_rejected():
    with pytest.raises(ValueError, match="got event for"):
        EventCalendar("TEST", [Event("OTHER", pd.Timestamp("2026-01-08 16:00", tz="US/Eastern"),
                                      EventType.EARNINGS)])


def test_future_dates_not_implemented():
    cal = EventCalendar("TEST", [])
    with pytest.raises(NotImplementedError):
        cal.horizon_event_schedule(_ts(), start_idx=1, horizon_days=2,
                                   future_dates=np.array([1, 2]))


# ---------- forward_schedule (live forward projection) ----------

from datetime import date  # noqa: E402


def _nth_business_day_after(run: date, n: int) -> str:
    """Date of the n-th business day strictly after `run` (run assumed a business day)."""
    return pd.Timestamp(np.busday_offset(np.datetime64(run), n)).date().isoformat()


def test_forward_schedule_empty_is_all_none():
    cal = EventCalendar("TEST", [])
    assert cal.forward_schedule(date(2026, 1, 5), horizon_days=5) == [None] * 5


def test_forward_schedule_amc_earnings_lands_on_next_session():
    run = date(2026, 1, 5)  # Monday
    f1 = _nth_business_day_after(run, 2)  # 2nd business day after run (horizon day 1's date)
    cal = EventCalendar("TEST", [_event(f1, EventTiming.AMC)])
    # AMC on F1 → move realised on the NEXT session → horizon day 2.
    assert cal.forward_schedule(run, horizon_days=5) == [None, None, EventType.EARNINGS, None, None]


def test_forward_schedule_bmo_earnings_same_day():
    run = date(2026, 1, 5)
    f2 = _nth_business_day_after(run, 3)  # 3rd business day after run
    cal = EventCalendar("TEST", [_event(f2, EventTiming.BMO, hour=8)])
    # BMO on F2 → same day's return → horizon day 2.
    assert cal.forward_schedule(run, horizon_days=5) == [None, None, EventType.EARNINGS, None, None]


def test_forward_schedule_amc_today_hits_first_horizon_day():
    run = date(2026, 1, 5)  # Monday, a business day
    cal = EventCalendar("TEST", [_event(run.isoformat(), EventTiming.AMC)])
    # Earnings AMC tonight → tomorrow's gap → horizon day 0.
    assert cal.forward_schedule(run, horizon_days=5)[0] == EventType.EARNINGS


def test_forward_schedule_event_beyond_horizon_dropped():
    run = date(2026, 1, 5)
    far = _nth_business_day_after(run, 30)  # well past a 5-day horizon
    cal = EventCalendar("TEST", [_event(far, EventTiming.AMC)])
    assert cal.forward_schedule(run, horizon_days=5) == [None] * 5


def test_forward_schedule_precedence_on_collision():
    run = date(2026, 1, 5)
    f1 = _nth_business_day_after(run, 2)
    cal = EventCalendar("TEST", [
        _event(f1, EventTiming.AMC, EventType.MACRO_FOMC),
        _event(f1, EventTiming.AMC, EventType.EARNINGS),
    ])
    sched = cal.forward_schedule(run, horizon_days=5)
    assert sched[2] == EventType.EARNINGS  # EARNINGS outranks MACRO_FOMC


def test_forward_schedule_bad_horizon_raises():
    cal = EventCalendar("TEST", [])
    with pytest.raises(ValueError, match="horizon_days"):
        cal.forward_schedule(date(2026, 1, 5), horizon_days=0)
