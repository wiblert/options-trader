"""Tests for the Event dataclass and its enums."""

import pandas as pd
import pytest

from options_trader.data.events.event import Event, EventTiming, EventType


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="US/Eastern")


def test_event_type_is_str_enum():
    assert EventType.EARNINGS == "earnings"
    assert EventType("earnings") is EventType.EARNINGS


def test_event_type_extensible_membership():
    # All declared members round-trip from their string value.
    for member in EventType:
        assert EventType(member.value) is member


def test_construct_valid_event():
    e = Event("META", _ts("2026-04-29 16:00"), EventType.EARNINGS, EventTiming.AMC)
    assert e.ticker == "META"
    assert e.event_type == EventType.EARNINGS
    assert e.timing == EventTiming.AMC


def test_string_type_and_timing_coerced():
    e = Event("AAPL", _ts("2026-05-01 08:00"), "earnings", "bmo")
    assert e.event_type is EventType.EARNINGS
    assert e.timing is EventTiming.BMO


def test_default_timing_is_unknown():
    e = Event("AAPL", _ts("2026-05-01 12:00"), EventType.EARNINGS)
    assert e.timing == EventTiming.UNKNOWN


def test_empty_ticker_raises():
    with pytest.raises(ValueError, match="ticker"):
        Event("", _ts("2026-05-01"), EventType.EARNINGS)


def test_naive_timestamp_raises():
    with pytest.raises(ValueError, match="tz-aware"):
        Event("AAPL", pd.Timestamp("2026-05-01 16:00"), EventType.EARNINGS)


def test_non_timestamp_raises():
    with pytest.raises(TypeError, match="Timestamp"):
        Event("AAPL", "2026-05-01", EventType.EARNINGS)


def test_event_is_frozen():
    e = Event("AAPL", _ts("2026-05-01 16:00"), EventType.EARNINGS)
    with pytest.raises(Exception):
        e.ticker = "MSFT"  # type: ignore[misc]


def test_metadata_defaults_independent():
    a = Event("AAPL", _ts("2026-05-01 16:00"), EventType.EARNINGS)
    b = Event("MSFT", _ts("2026-05-02 16:00"), EventType.EARNINGS)
    a.metadata["x"] = 1
    assert b.metadata == {}
