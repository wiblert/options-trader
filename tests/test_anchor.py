"""Tests for the injectable spot-anchor sources and the anchored rolling backtest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_trader.backtest.anchor import (
    ANCHOR_FNS,
    MA_BLEND_ANCHOR_FNS,
    AnchorContext,
    intraday_uniform_anchor,
    make_ma_blend_anchor,
    ma7_blend_anchor,
    stale_close_anchor,
)
from options_trader.backtest.log_score import rolling_log_score_backtest
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.return_distribution import ReturnDistribution


def _ctx(rng=None) -> AnchorContext:
    return AnchorContext(
        prev_close=100.0,
        today_open=101.0,
        today_high=104.0,
        today_low=99.0,
        today_close=102.0,
        rng=rng or np.random.default_rng(0),
    )


def _ohlc_ts(n: int = 300, seed: int = 0, ticker: str = "TEST") -> StockReturnTS:
    """Synthetic GBM with a genuine intraday OHLC range around each close."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.015, size=n - 1)
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], log_returns])))
    # A real intraday range: high/low straddle the close by ~1%.
    span = close * 0.01
    high = close + span
    low = close - span
    open_ = close - 0.5 * span
    dates = pd.date_range("2024-01-02", periods=n, freq="B").to_numpy()
    return StockReturnTS(
        ticker=ticker, dates=dates,
        open=open_, high=high, low=low, close=close,
        volume=np.full(n, 1e6), source="alpaca",
        start=date(2024, 1, 1), end=date(2026, 1, 1),
    )


def _factory(rd: ReturnDistribution) -> BootstrapForecaster:
    return BootstrapForecaster(rd, n_paths=3_000, seed=7)


# ---------- anchor functions ----------

def test_stale_close_returns_prev_close():
    assert stale_close_anchor(_ctx()) == 100.0


def test_intraday_uniform_within_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = intraday_uniform_anchor(_ctx(rng))
        assert 99.0 <= p <= 104.0


def test_intraday_uniform_reproducible_and_varies():
    a = intraday_uniform_anchor(_ctx(np.random.default_rng(42)))
    b = intraday_uniform_anchor(_ctx(np.random.default_rng(42)))
    assert a == b  # same seed -> same draw
    # advancing the same generator yields a different draw
    rng = np.random.default_rng(42)
    assert intraday_uniform_anchor(_ctx(rng)) != intraday_uniform_anchor(_ctx(rng))


def test_registry_keys():
    assert set(ANCHOR_FNS) == {"stale_close", "intraday_random"}


# ---------- moving-average blend anchors (Hypothesis 1) ----------

def _ctx_with_closes(closes, rng=None) -> AnchorContext:
    closes = np.asarray(closes, dtype=float)
    return AnchorContext(
        prev_close=float(closes[-1]),
        today_open=101.0, today_high=104.0, today_low=99.0, today_close=102.0,
        rng=rng or np.random.default_rng(0),
        trailing_closes=closes,
    )


def test_ma7_blend_is_half_latest_half_ma7():
    closes = np.arange(1, 11, dtype=float)  # MA7 over last 7 = mean(4..10) = 7.0
    # Pin the fresh draw by using the same generator the anchor will draw from.
    rng = np.random.default_rng(0)
    latest = np.random.default_rng(0).uniform(99.0, 104.0)  # what the anchor will draw
    got = ma7_blend_anchor(_ctx_with_closes(closes, rng))
    assert got == pytest.approx(0.5 * latest + 0.5 * 7.0)


def test_ma_blend_uses_only_trailing_window_point_in_time():
    """The MA must come from trailing_closes only — not today_close/future."""
    closes = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])  # MA7 = 40.0
    anchor = make_ma_blend_anchor(window=7, weight_latest=0.0)  # pure MA
    # today_close=102.0 in the ctx must be ignored.
    assert anchor(_ctx_with_closes(closes)) == pytest.approx(40.0)


def test_ma_only_ignores_latest_but_still_advances_rng():
    closes = np.full(7, 50.0)  # MA = 50.0
    pure_ma = make_ma_blend_anchor(window=7, weight_latest=0.0)
    rng = np.random.default_rng(5)
    assert pure_ma(_ctx_with_closes(closes, rng)) == pytest.approx(50.0)
    # one uniform consumed → generator advanced (CRN parity with intraday arm)
    assert rng.uniform(0, 1) != np.random.default_rng(5).uniform(0, 1)


def test_ma_blend_shorter_history_uses_available():
    closes = np.array([2.0, 4.0])  # only 2 closes, window 7 → mean(2,4)=3.0
    pure_ma = make_ma_blend_anchor(window=7, weight_latest=0.0)
    assert pure_ma(_ctx_with_closes(closes)) == pytest.approx(3.0)


def test_ma_blend_requires_trailing_closes():
    with pytest.raises(ValueError, match="trailing_closes"):
        ma7_blend_anchor(_ctx())  # default ctx has empty trailing_closes


def test_make_ma_blend_validates_args():
    with pytest.raises(ValueError, match="weight_latest"):
        make_ma_blend_anchor(window=7, weight_latest=1.5)
    with pytest.raises(ValueError, match="window"):
        make_ma_blend_anchor(window=0, weight_latest=0.5)


def test_ma_blend_registry_keys():
    assert set(MA_BLEND_ANCHOR_FNS) == {
        "intraday_random", "ma7_blend", "ma20_blend", "ma7_only"
    }


def test_ma7_blend_equals_half_intraday_plus_half_ma_in_backtest():
    """End-to-end: with a shared anchor_seed, ma7_blend's spot at each t equals
    0.5*intraday_random spot + 0.5*(trailing 7-close MA through t) — confirming
    both CRN alignment and point-in-time MA."""
    ts = _ohlc_ts(n=200)
    kw = dict(horizon_days=5, holdout_days=40, anchor_seed=99)
    intraday = rolling_log_score_backtest(
        ts, _factory, anchor_fn=intraday_uniform_anchor, **kw
    )
    blend = rolling_log_score_backtest(
        ts, _factory, anchor_fn=ma7_blend_anchor, **kw
    )
    close = np.asarray(ts.close, dtype=float)
    T = len(close) - 40 - 1
    for e_i, e_b, t in zip(intraday.evaluations, blend.evaluations, range(T, len(close) - 5)):
        ma7 = float(np.mean(close[: t + 1][-7:]))
        assert e_b.spot == pytest.approx(0.5 * e_i.spot + 0.5 * ma7)


# ---------- anchored rolling backtest ----------

def test_default_path_anchors_on_close():
    """No anchor_fn -> spot is exactly close[t] (historical behaviour preserved)."""
    ts = _ohlc_ts(n=200)
    res = rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=40)
    T = len(ts.close) - 40 - 1
    expected = [float(ts.close[t]) for t in range(T, len(ts.close) - 5)]
    np.testing.assert_array_equal([e.spot for e in res.evaluations], expected)


def test_stale_anchor_matches_default():
    """stale_close anchors on close[t] -> identical spots to the default path."""
    ts = _ohlc_ts(n=200)
    base = rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=40)
    stale = rolling_log_score_backtest(
        ts, _factory, horizon_days=5, holdout_days=40,
        anchor_fn=stale_close_anchor,
    )
    np.testing.assert_array_equal(
        [e.spot for e in base.evaluations], [e.spot for e in stale.evaluations]
    )


def test_intraday_anchor_spots_in_next_day_range():
    """intraday_random draws each spot from the NEXT day's [low, high]."""
    ts = _ohlc_ts(n=200)
    res = rolling_log_score_backtest(
        ts, _factory, horizon_days=5, holdout_days=40,
        anchor_fn=intraday_uniform_anchor,
    )
    T = len(ts.close) - 40 - 1
    for e, t in zip(res.evaluations, range(T, len(ts.close) - 5)):
        assert ts.low[t + 1] <= e.spot <= ts.high[t + 1]


def test_intraday_anchor_reproducible():
    ts = _ohlc_ts(n=200)
    kw = dict(horizon_days=5, holdout_days=40, anchor_fn=intraday_uniform_anchor, anchor_seed=99)
    a = rolling_log_score_backtest(ts, _factory, **kw)
    b = rolling_log_score_backtest(ts, _factory, **kw)
    np.testing.assert_array_equal(
        [e.spot for e in a.evaluations], [e.spot for e in b.evaluations]
    )


def test_anchor_requires_horizon_at_least_two():
    ts = _ohlc_ts(n=200)
    with pytest.raises(ValueError, match="horizon_days >= 2"):
        rolling_log_score_backtest(
            ts, _factory, horizon_days=1, holdout_days=40,
            anchor_fn=intraday_uniform_anchor,
        )
