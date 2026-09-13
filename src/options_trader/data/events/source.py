"""
source.py — the EventSource abstraction.

EventSource is a swappable backend that fetches events for a ticker over a window.
The window's `end` MAY be in the future — a source is expected to return both
historical events (for building conditioned distributions) and upcoming events (for
scheduling the forecast horizon).

It's a structural typing.Protocol rather than an ABC so backends don't inherit
anything — mirrors how the rest of the codebase passes plain callables/data bags.
V1 backends: YFinanceEarningsSource, ManualCalendarSource, CompositeEventSource.
Future: AlpacaNewsSource, etc.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from options_trader.data.events.event import Event


class EventSourceError(Exception):
    """Raised when an event source cannot fetch/parse events. Callers may choose
    to degrade to 'no events' (which makes conditioning a no-op) on this error."""


@runtime_checkable
class EventSource(Protocol):
    """A backend that returns events for a ticker over a (possibly future) window."""

    def fetch_events(self, ticker: str, start: date, end: date) -> list[Event]:
        """Return all events for `ticker` with timestamp in [start, end).

        `end` may be in the future to capture upcoming events. The returned list
        is sorted ascending by timestamp.
        """
        ...
