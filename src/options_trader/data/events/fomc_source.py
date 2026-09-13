"""
fomc_source.py — FOMC (Fed rate-decision) meeting dates from the Federal Reserve.

An EventSource that scrapes the Fed's official FOMC calendar page. Unlike the manual
CSV (which goes stale the moment the hand-entered dates run out), this auto-updates:
the page carries ~6 years of meetings — recent history AND scheduled future — so a
single fetch serves both the historical conditioning distribution and the forward
schedule.

FOMC is a MARKET-WIDE event, so (like the manual calendar's '*' wildcard) every event
is tagged with the queried `ticker`. Per-ticker conditioning then captures that
ticker's own behaviour on Fed days.

Parsing target (Fed HTML): year panels
    <h4><a id="...">2026 FOMC Meetings</a></h4>
followed by per-meeting rows
    <div class="fomc-meeting__month"><strong>January</strong></div>
    <div class="fomc-meeting__date">27-28</div>
The announcement (rate decision) is the LAST day of the meeting; we take the last day
number. Edge cases handled: cross-month meetings ("Apr/May" + "30-1" -> May 1),
SEP-meeting markers ("17-18*"), and unscheduled "notation vote" rows (excluded —
not a scheduled press-conference decision).

The pure `_parse_fomc_calendar(html)` core is unit-tested on a saved HTML sample with
no network (mirrors yfinance_source._rows_to_events / history.py).
"""

from __future__ import annotations

import logging
import re
from datetime import date

import pandas as pd
import requests

from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.events.source import EventSourceError


logger = logging.getLogger(__name__)


FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
# Fed servers reject the default urllib UA (mirrors universe/snapshot.py).
_BROWSER_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
_HTTP_TIMEOUT_S = 20.0

# Statement release time (rate decision is announced ~2:00pm ET on the last meeting day).
_ANNOUNCE_HOUR_ET = 14

_YEAR_RE = re.compile(r'<a id="\d+">\s*(\d{4})\s+FOMC Meetings\s*</a>')
_MONTH_RE = re.compile(r'fomc-meeting__month[^>]*>(.*?)</div>', re.S)
_DATE_RE = re.compile(r'fomc-meeting__date[^>]*>(.*?)</div>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")

# First three lowercase letters of a month name -> month number (handles both full
# names "September" and the Fed's cross-month abbreviations "Apr/May", "Jan/Feb").
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _clean(text: str) -> str:
    return _TAG_RE.sub("", text).replace("\xa0", " ").strip()


def _parse_fomc_calendar(html: str) -> list[date]:
    """Parse FOMC announcement (rate-decision) dates from the Fed calendar HTML.

    Returns sorted, de-duplicated dates (the LAST day of each scheduled meeting).
    Pure: no network. Returns [] if the structure yields no meetings (the caller
    treats that as a parse failure on a non-empty page).
    """
    headings = list(_YEAR_RE.finditer(html))
    out: set[date] = set()
    for i, h in enumerate(headings):
        year = int(h.group(1))
        seg_end = headings[i + 1].start() if i + 1 < len(headings) else len(html)
        segment = html[h.end():seg_end]

        months = [_clean(m) for m in _MONTH_RE.findall(segment)]
        dates = [_clean(d) for d in _DATE_RE.findall(segment)]
        if len(months) != len(dates):
            logger.warning("FOMC parse: %d month vs %d date cells in %d panel; zipping the overlap",
                           len(months), len(dates), year)
        for month_field, date_field in zip(months, dates):
            if "notation" in date_field.lower():
                continue  # unscheduled procedural vote, not a press-conference decision
            day_nums = re.findall(r"\d+", date_field)
            if not day_nums:
                continue
            last_day = int(day_nums[-1])
            # Cross-month meeting ("Apr/May") -> the last day belongs to the 2nd month.
            month_token = month_field.split("/")[-1].strip()[:3].lower()
            month = _MONTHS.get(month_token)
            if month is None:
                continue
            try:
                out.add(date(year, month, last_day))
            except ValueError:
                logger.warning("FOMC parse: bad date %d-%s-%d", year, month_token, last_day)
    return sorted(out)


class FomcCalendarSource:
    """EventSource backend: FOMC meeting dates scraped from the Fed (auto-updating)."""

    def __init__(
        self,
        timing: EventTiming = EventTiming.BMO,
        url: str = FOMC_CALENDAR_URL,
        timeout: float = _HTTP_TIMEOUT_S,
    ) -> None:
        """
        Args:
            timing: which session's return the decision attaches to (see
                calendar.py's mapping). The statement drops ~2:00pm ET, so the move
                lands in the announcement DAY's own close-to-close return — that is
                the BMO branch (index q-1 = same-day return), NOT the literal
                before-open meaning. Default BMO; flip to AMC (next-day) and backtest
                if that scores better. (Conditioning + forward schedule both use this,
                so they stay internally consistent either way.)
            url / timeout: override for testing.
        """
        self.timing = timing
        self.url = url
        self.timeout = timeout

    def _get(self) -> str:
        """Fetch the calendar HTML. Isolated so tests can stub it without network."""
        resp = requests.get(self.url, headers=_BROWSER_UA, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def fetch_events(self, ticker: str, start: date, end: date) -> list[Event]:
        try:
            html = self._get()
        except Exception as exc:  # noqa: BLE001 — requests raises a zoo of types
            raise EventSourceError(f"FOMC calendar fetch failed: {exc!r}") from exc

        dates = _parse_fomc_calendar(html)
        if not dates:
            raise EventSourceError(
                f"FOMC calendar parsed 0 meetings from {self.url} (page structure may have changed)"
            )

        events: list[Event] = []
        for d in dates:
            if not (start <= d < end):
                continue
            ts = pd.Timestamp(f"{d.isoformat()} {_ANNOUNCE_HOUR_ET:02d}:00", tz="US/Eastern")
            events.append(Event(
                ticker=ticker,
                timestamp=ts,
                event_type=EventType.MACRO_FOMC,
                timing=self.timing,
                metadata={"source": "fomc_calendar"},
            ))
        events.sort(key=lambda e: e.timestamp)
        logger.info("FomcCalendarSource: %d FOMC events for %s in [%s, %s)",
                    len(events), ticker, start, end)
        return events
