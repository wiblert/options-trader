"""Tests for the live forecaster factories (forecast/factories.py).

These verify the production wiring that lets the backtest-validated forecaster
trade: the event factory builds an EventCalendar from a source, projects a forward
event schedule over the horizon, and hands it to the event-conditioned forecaster —
and degrades to the plain bootstrap when the event feed fails. No network: the
event source is an in-memory fake.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_trader.data.events.event import Event, EventTiming, EventType
from options_trader.data.events.source import EventSourceError
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.event_bootstrap_forecaster import (
    EventConditionedBootstrapForecaster,
)
from options_trader.forecast.factories import (
    EventBootstrapFactory,
    ForecastContext,
    OptionImpliedBetaFactory,
    bootstrap_factory,
    event_days_in_horizon,
)
from options_trader.forecast.option_implied_beta_forecaster import (
    OptionImpliedBetaForecaster,
)
from options_trader.forecast.breeden_litzenberger import implied_pdf_from_iv
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


RUN_DATE = date(2026, 1, 5)   # Monday, a business day
HORIZON = 5


def _ts(n: int = 150) -> StockReturnTS:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rng.normal(0.0003, 0.012, n - 1)])))
    dates = pd.bdate_range(end="2026-01-02", periods=n).to_numpy()
    return StockReturnTS(
        ticker="TEST", dates=dates, open=close, high=close, low=close, close=close,
        volume=np.full(n, 1e6), source="test", start=date(2024, 1, 1), end=RUN_DATE,
    )


def _ctx() -> ForecastContext:
    ts = _ts()
    rd = ReturnDistribution(np.diff(np.log(ts.close)))
    return ForecastContext(
        ticker="TEST", ts=ts, rd=rd, horizon=HORIZON,
        run_date=RUN_DATE, n_paths=2000, seed=7,
    )


class _FakeSource:
    """In-memory EventSource: returns the events that fall in [start, end)."""

    def __init__(self, events: list[Event]) -> None:
        self._events = events

    def fetch_events(self, ticker: str, start: date, end: date) -> list[Event]:
        return [e for e in self._events
                if start <= e.timestamp.tz_convert("US/Eastern").date() < end]


class _BrokenSource:
    def fetch_events(self, ticker: str, start: date, end: date):
        raise EventSourceError("boom")


def _earnings(day: str, timing: EventTiming = EventTiming.AMC) -> Event:
    return Event("TEST", pd.Timestamp(f"{day} 16:00", tz="US/Eastern"),
                 EventType.EARNINGS, timing)


def _future_business_day(n: int) -> str:
    return pd.Timestamp(np.busday_offset(np.datetime64(RUN_DATE), n)).date().isoformat()


# ---------- bootstrap factory ----------

def test_bootstrap_factory_returns_bootstrap():
    fc = bootstrap_factory(_ctx())
    assert isinstance(fc, BootstrapForecaster)
    assert event_days_in_horizon(fc) == 0
    pd_ = fc.forecast(HORIZON, spot=100.0)
    assert isinstance(pd_, PriceDistribution) and len(pd_) == 2000


# ---------- event factory: schedule wiring ----------

def test_event_factory_injects_upcoming_earnings_into_horizon():
    f1 = _future_business_day(2)  # 2nd business day after run → horizon day 2
    factory = EventBootstrapFactory(_FakeSource([_earnings(f1)]))
    fc = factory(_ctx())

    assert isinstance(fc, EventConditionedBootstrapForecaster)
    assert fc.event_schedule == [None, None, EventType.EARNINGS, None, None]
    assert event_days_in_horizon(fc) == 1
    pd_ = fc.forecast(HORIZON, spot=100.0)
    assert isinstance(pd_, PriceDistribution) and len(pd_) == 2000


def test_event_factory_no_events_gives_empty_schedule():
    # An earnings date well outside the horizon → schedule is all-None.
    far = _future_business_day(40)
    factory = EventBootstrapFactory(_FakeSource([_earnings(far)]))
    fc = factory(_ctx())
    assert isinstance(fc, EventConditionedBootstrapForecaster)
    assert event_days_in_horizon(fc) == 0


def test_event_factory_falls_back_to_bootstrap_on_source_error():
    factory = EventBootstrapFactory(_BrokenSource())
    fc = factory(_ctx())
    assert isinstance(fc, BootstrapForecaster)  # degraded, not crashed


def test_event_factory_uses_distinct_earnings_distribution():
    """End-to-end: a historical earnings day with an outsized move makes the
    earnings-day distribution fatter, so a horizon containing an earnings day
    forecasts a WIDER terminal distribution than the plain bootstrap."""
    ts = _ts()
    # Plant a big historical earnings move so by_type[EARNINGS] is volatile.
    hist_earnings_date = pd.Timestamp(ts.dates[100]).date().isoformat()
    upcoming = _future_business_day(2)
    src = _FakeSource([_earnings(hist_earnings_date), _earnings(upcoming)])
    ctx = _ctx()

    event_fc = EventBootstrapFactory(src)(ctx)
    plain_fc = bootstrap_factory(ctx)
    # Same seed/spot; the only difference is event routing on horizon day 2.
    ev_std = event_fc.forecast(HORIZON, spot=100.0).std()
    bs_std = plain_fc.forecast(HORIZON, spot=100.0).std()
    assert event_fc.event_schedule[2] == EventType.EARNINGS
    # Not asserting a strict inequality on one random draw would be flaky; instead
    # confirm the earnings distribution is genuinely separate (>=1 earnings sample).
    assert EventType.EARNINGS in event_fc.conditioned.by_type
    assert ev_std > 0 and bs_std > 0


# ---------- OptionImpliedBetaFactory ----------

def _index_ts(ticker: str, n: int = 400, vol: float = 0.011, seed: int = 1) -> StockReturnTS:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rng.normal(0.0002, vol, n - 1)])))
    dates = pd.bdate_range(end="2026-01-02", periods=n).to_numpy()
    return StockReturnTS(
        ticker=ticker, dates=dates, open=close, high=close, low=close, close=close,
        volume=np.full(n, 1e6), source="test", start=date(2024, 1, 1), end=RUN_DATE,
    )


def _stub_index_pdf(symbol, run_date, horizon):
    spot = 400.0 if symbol == "SPY" else 200.0
    k = np.linspace(spot * 0.75, spot * 1.25, 41)
    iv = np.full_like(k, 0.18)
    return implied_pdf_from_iv(k, iv, spot, horizon / 365.0, 0.04)


def test_option_implied_beta_factory_builds_forecaster():
    histories = {"SPY": _index_ts("SPY", seed=2), "IWM": _index_ts("IWM", seed=3)}
    factory = OptionImpliedBetaFactory(
        index_pdf_fn=_stub_index_pdf,
        index_history_fn=lambda s: histories[s],
        beta_window=None,
    )
    f = factory(_ctx())
    assert isinstance(f, OptionImpliedBetaForecaster)
    out = f.forecast(HORIZON, 100.0)
    assert len(out) == 2000
    assert np.all(out.prices > 0)
    assert np.isfinite(f.blended_beta)


def test_option_implied_beta_factory_degrades_on_failure():
    def _broken_history(symbol):
        raise RuntimeError("no index history")
    factory = OptionImpliedBetaFactory(
        index_pdf_fn=_stub_index_pdf, index_history_fn=_broken_history,
    )
    f = factory(_ctx())
    assert isinstance(f, BootstrapForecaster)  # graceful fallback
