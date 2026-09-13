"""Tests for the Option-Implied Beta backtest plumbing.

Synthetic / no-network: a stub TradingClient serves an expired BS-priced chain and
a stub OptionBarsProvider serves matching EOD bars, so we can verify the historical
index-PDF builder and the rolling backtest end to end without hitting Alpaca.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from options_trader.data.options_history import OptionBar
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.backtest.option_implied_beta_backtest import (
    HistoricalIndexPdf,
    _spot_lookup_from_ts,
    _truncate_ts_by_date,
    build_eod_index_pdf,
    rolling_log_score_backtest_option_implied_beta,
)
from options_trader.valuation.black_scholes import bs_price


RUN_DATE = date(2025, 3, 17)
EXPIRY = date(2025, 4, 17)
SIG = 0.20
R = 0.04


def _occ(sym, strike, ctype):
    cp = "C" if ctype == "call" else "P"
    return f"{sym}{EXPIRY:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


class _StubTradingClient:
    """Serves a flat-vol BS chain of expired contracts at one expiry."""

    def __init__(self, underlying, spot, n=61):
        self.underlying = underlying
        self.strikes = np.linspace(spot * 0.80, spot * 1.20, n)

    def get_option_contracts(self, req):
        ctype = req.type.value if req.type is not None else "call"
        lo = float(req.strike_price_gte) if req.strike_price_gte else -np.inf
        hi = float(req.strike_price_lte) if req.strike_price_lte else np.inf
        contracts = [
            SimpleNamespace(symbol=_occ(self.underlying, k, ctype),
                            strike_price=str(k), expiration_date=EXPIRY, type=ctype)
            for k in self.strikes if lo <= k <= hi
        ]
        return SimpleNamespace(option_contracts=contracts, next_page_token=None)


class _StubBarsProvider:
    """EOD bars priced off the same flat-vol BS world, dated exactly RUN_DATE."""

    def __init__(self, spot, t_years):
        self.spot = spot
        self.t = t_years

    def get_eod_bars(self, symbols, start, end):
        out = {}
        for s in symbols:
            cp = s[-9]  # 'C' or 'P'
            strike = int(s[-8:]) / 1000.0
            otype = "call" if cp == "C" else "put"
            px = bs_price(self.spot, strike, self.t, R, SIG, otype)
            out[s] = [OptionBar(s, RUN_DATE, px, px, px, px, 500.0)]
        return out


def test_build_eod_index_pdf_recovers_lognormal():
    spot = 560.0
    t = (EXPIRY - RUN_DATE).days / 365.0
    pdf = build_eod_index_pdf(
        "SPY", RUN_DATE, horizon_days=21, spot=spot,
        trading_client=_StubTradingClient("SPY", spot),
        bars_provider=_StubBarsProvider(spot, t), risk_free_rate=R,
    )
    fwd = spot * np.exp(R * t)
    ln_std = spot * np.exp(R * t) * np.sqrt(np.exp(SIG ** 2 * t) - 1)
    assert pdf.mean_price() == pytest.approx(fwd, rel=0.01)
    assert pdf.std_price() == pytest.approx(ln_std, rel=0.10)


def test_build_eod_index_pdf_raises_when_no_contracts():
    class _Empty:
        def get_option_contracts(self, req):
            return SimpleNamespace(option_contracts=[], next_page_token=None)
    with pytest.raises(ValueError):
        build_eod_index_pdf("SPY", RUN_DATE, 21, 560.0,
                            trading_client=_Empty(), bars_provider=_StubBarsProvider(560.0, 0.08))


def _price_path(returns, p0=100.0):
    return p0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))


def _ts(ticker, returns, p0, start="2024-01-01"):
    closes = _price_path(returns, p0)
    n = len(closes)
    dates = pd.bdate_range(start, periods=n).to_numpy()
    return StockReturnTS(ticker=ticker, dates=dates, open=closes, high=closes,
                         low=closes, close=closes, volume=np.full(n, 1e6),
                         source="synthetic", start=None, end=None)


def test_truncate_ts_by_date_is_point_in_time():
    ts = _ts("X", np.full(100, 0.001), 100.0, start="2024-01-01")
    cut = pd.Timestamp(ts.dates[50]).date()
    tr = _truncate_ts_by_date(ts, cut)
    assert pd.Timestamp(tr.dates[-1]).date() <= cut
    assert len(tr) == 51


def test_spot_lookup_uses_on_or_before():
    ts = _ts("SPY", np.full(30, 0.0), 400.0, start="2024-01-01")
    lookup = _spot_lookup_from_ts(ts)
    # A weekend date between bars resolves to the most recent prior close.
    d = pd.Timestamp(ts.dates[10]).date()
    assert lookup("SPY", d) == pytest.approx(float(ts.close[10]))


def test_backtest_runs_with_stub_index_pdf():
    rng = np.random.default_rng(0)
    n = 320
    spy_r = rng.normal(0.0002, 0.009, n)
    iwm_r = rng.normal(0.0002, 0.011, n)
    stock_r = 1.1 * spy_r + 0.6 * iwm_r + rng.normal(0, 0.004, n)
    spy = _ts("SPY", spy_r, 400.0); iwm = _ts("IWM", iwm_r, 200.0)
    stock = _ts("XYZ", stock_r, 100.0)

    # Index PDF stub: a fixed lognormal return distribution per index, ignoring date.
    from options_trader.forecast.breeden_litzenberger import implied_pdf_from_iv

    def stub_pdf(symbol, run_date, horizon):
        s = 400.0 if symbol == "SPY" else 200.0
        k = np.linspace(s * 0.8, s * 1.2, 41)
        return implied_pdf_from_iv(k, np.full_like(k, 0.18), s, horizon / 365.0, R)

    res = rolling_log_score_backtest_option_implied_beta(
        stock, spy, iwm, stub_pdf, horizon_days=21, holdout_days=30,
        n_paths=3000, beta_window=None,
    )
    assert res.n == 30 - 21 + 1
    assert res.config["n_skipped"] == 0
    assert np.isfinite(res.mean_log_score)


def test_backtest_skips_dates_when_pdf_unavailable():
    rng = np.random.default_rng(1)
    n = 320
    spy = _ts("SPY", rng.normal(0, 0.009, n), 400.0)
    iwm = _ts("IWM", rng.normal(0, 0.011, n), 200.0)
    stock = _ts("XYZ", rng.normal(0, 0.012, n), 100.0)

    def broken_pdf(symbol, run_date, horizon):
        raise ValueError("no chain")

    res = rolling_log_score_backtest_option_implied_beta(
        stock, spy, iwm, broken_pdf, horizon_days=21, holdout_days=30, n_paths=2000,
    )
    assert res.n == 0
    assert res.config["n_skipped"] == 30 - 21 + 1


def test_conditioned_residuals_captures_earnings():
    """Earnings-day residuals (a planted down-jump) form a distinct distribution
    with a clearly negative mean vs the ~zero-mean normal residuals."""
    from options_trader.data.events.event import Event, EventType, EventTiming
    from options_trader.data.events.calendar import EventCalendar
    from options_trader.backtest.option_implied_beta_backtest import _conditioned_residuals

    rng = np.random.default_rng(0)
    n = 400
    spy_r = rng.normal(0, 0.009, n - 1)
    iwm_r = rng.normal(0, 0.011, n - 1)
    eps = rng.normal(0, 0.004, n - 1)
    earn = [80, 160, 240, 320]
    for j in earn:
        eps[j] = -0.10                      # planted earnings down-jumps
    stock_r = 1.0 * (0.5 * spy_r + 0.5 * iwm_r) + eps
    S = _ts("XYZ", stock_r, 100.0); SPY = _ts("SPY", spy_r, 400.0); IWM = _ts("IWM", iwm_r, 200.0)
    # BMO event on S.dates[j+1] maps to return index j (use the SAME dates as S).
    evs = [Event("XYZ", pd.Timestamp(S.dates[j + 1]).tz_localize("US/Eastern") + pd.Timedelta(hours=8),
                 EventType.EARNINGS, EventTiming.BMO) for j in earn]
    cal = EventCalendar("XYZ", evs)

    b_spy, b_iwm, normal_rd, idio_ev = _conditioned_residuals(S, SPY, IWM, cal, beta_window=None)
    assert EventType.EARNINGS in idio_ev
    earn_mean = float(np.sum(idio_ev[EventType.EARNINGS].samples * idio_ev[EventType.EARNINGS].weights))
    norm_mean = float(np.sum(normal_rd.samples * normal_rd.weights))
    assert earn_mean < -0.05            # captured the down-jump
    assert abs(norm_mean) < 0.01        # normal residuals ~ zero-mean


def test_event_backtest_runs_with_stubs():
    from options_trader.data.events.event import Event, EventType, EventTiming
    from options_trader.data.events.calendar import EventCalendar
    from options_trader.forecast.breeden_litzenberger import implied_pdf_from_iv
    from options_trader.backtest.option_implied_beta_backtest import (
        rolling_log_score_backtest_option_implied_beta_event)

    rng = np.random.default_rng(1)
    n = 320
    spy = _ts("SPY", rng.normal(0, 0.009, n), 400.0)
    iwm = _ts("IWM", rng.normal(0, 0.011, n), 200.0)
    stock = _ts("XYZ", rng.normal(0.0002, 0.012, n), 100.0)
    # An earnings event inside the holdout horizon window.
    ev_date = pd.Timestamp(stock.dates[-15]).tz_localize("US/Eastern") + pd.Timedelta(hours=8)
    cal = EventCalendar("XYZ", [Event("XYZ", ev_date, EventType.EARNINGS, EventTiming.BMO)])

    def stub_pdf(symbol, run_date, horizon):
        s = 400.0 if symbol == "SPY" else 200.0
        k = np.linspace(s * 0.8, s * 1.2, 41)
        return implied_pdf_from_iv(k, np.full_like(k, 0.18), s, horizon / 365.0, R)

    res = rolling_log_score_backtest_option_implied_beta_event(
        stock, spy, iwm, stub_pdf, cal, horizon_days=21, holdout_days=30,
        n_paths=3000, beta_window=None)
    assert res.n == 30 - 21 + 1
    assert np.isfinite(res.mean_log_score)


def test_historical_index_pdf_caches():
    spot = 560.0
    t = (EXPIRY - RUN_DATE).days / 365.0
    calls = {"n": 0}
    tc = _StubTradingClient("SPY", spot)
    orig = tc.get_option_contracts
    def counting(req):
        calls["n"] += 1
        return orig(req)
    tc.get_option_contracts = counting

    src = HistoricalIndexPdf(lambda s, d: spot, trading_client=tc,
                             bars_provider=_StubBarsProvider(spot, t), risk_free_rate=R)
    a = src("SPY", RUN_DATE, 21)
    n_after_first = calls["n"]
    b = src("SPY", RUN_DATE, 21)  # cached → no new contract calls
    assert calls["n"] == n_after_first
    assert np.array_equal(a.density, b.density)
