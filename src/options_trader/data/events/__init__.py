"""
events — the event layer for event-conditioned forecasting.

Sources market events (earnings, macro, FDA, ...) for a ticker, maps them onto a
StockReturnTS's daily-return series, and exposes a per-horizon-day event schedule.
Consumed by forecast/conditioned_returns.py to build per-event-type return
distributions, and by the event-aware bootstrap forecaster to route each horizon
day's draw to the matching distribution.

Pieces:
    event.py          — Event dataclass + EventType / EventTiming enums
    source.py         — EventSource Protocol + EventSourceError
    yfinance_source.py— YFinanceEarningsSource (structured earnings, V1)
    manual_calendar.py— ManualCalendarSource (CSV of macro/FDA dates)
    composite.py      — CompositeEventSource (merge + dedup several sources)
    calendar.py       — EventCalendar (return-index mapping + horizon schedule)
"""

from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.events.source import EventSource, EventSourceError

__all__ = [
    "Event",
    "EventType",
    "EventTiming",
    "EventSource",
    "EventSourceError",
]
