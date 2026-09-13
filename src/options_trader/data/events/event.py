"""
event.py — the Event dataclass and its type/timing enums.

An Event is a single dated catalyst for one ticker (an earnings announcement, a
macro print, an FDA decision, ...). Events are matched by `event_type`: the
conditioning machinery routes "this horizon day is an earnings day" to the
distribution of past earnings-day returns, so the type is the join key.

`timing` (BMO/AMC) is first-class rather than metadata because it determines WHICH
daily return the price move attaches to: an after-close announcement moves the next
trading day's return, a before-open one moves the same day's. See
calendar.py:EventCalendar for the index mapping that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class EventType(str, Enum):
    """Catalyst category. str-subclassed so members compare equal to their string
    value (clean dict keys, CSV/JSON serialisation) while still giving enum
    validation and matchability. Add a member to support a new event class."""

    EARNINGS = "earnings"
    DIVIDEND_EX = "dividend_ex"
    MACRO_CPI = "macro_cpi"
    MACRO_FOMC = "macro_fomc"
    FDA = "fda"


class EventTiming(str, Enum):
    """When in the trading day the event fires, relative to the session.

    BMO — before market open: the move is realised on the SAME trading day's return.
    AMC — after market close:  the move is realised on the NEXT trading day's return.
    UNKNOWN — unknown; consumers treat as AMC (the large-cap norm) with a warning.
    """

    BMO = "bmo"
    AMC = "amc"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Event:
    """A single dated catalyst for one ticker.

    Attributes:
        ticker: symbol the event pertains to.
        timestamp: tz-aware pandas Timestamp of the event, as sourced.
        event_type: EventType — the match/join key for conditioning.
        timing: EventTiming — drives return-index attribution (see module docstring).
        metadata: free-form per-type extras (e.g. surprise_pct, eps_estimate, source).
    """

    ticker: str
    timestamp: pd.Timestamp
    event_type: EventType
    timing: EventTiming = EventTiming.UNKNOWN
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("Event.ticker must be a non-empty string")
        if not isinstance(self.timestamp, pd.Timestamp):
            raise TypeError(
                f"Event.timestamp must be a pandas Timestamp, got {type(self.timestamp).__name__}"
            )
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"Event.timestamp must be tz-aware, got naive {self.timestamp!r}"
            )
        # Coerce/validate enum membership (also accepts the raw string value).
        if not isinstance(self.event_type, EventType):
            object.__setattr__(self, "event_type", EventType(self.event_type))
        if not isinstance(self.timing, EventTiming):
            object.__setattr__(self, "timing", EventTiming(self.timing))
