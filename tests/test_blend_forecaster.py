"""Tests for the 50/50 blend forecaster + its backtest (synthetic, no network)."""

import numpy as np
import pandas as pd
import pytest

from options_trader.forecast.base import Forecaster
from options_trader.forecast.blend_forecaster import (
    BlendForecaster,
    blend_price_distributions,
)
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.data.events.calendar import EventCalendar
from options_trader.forecast.breeden_litzenberger import implied_pdf_from_iv
from options_trader.backtest.blend_backtest import rolling_log_score_backtest_blend_event


# ---------- the mixture helper ----------

def _pd(prices, weights=None):
    return PriceDistribution(np.asarray(prices, float), weights)


def test_mixture_weights_sum_to_one():
    a = _pd([90, 100, 110])
    b = _pd([200, 210, 220, 230])
    out = blend_price_distributions([a, b])
    assert out.weights.sum() == pytest.approx(1.0)
    assert len(out) == 7  # concatenated


def test_mixture_mean_is_weighted_average_of_component_means():
    a = _pd([90, 100, 110])      # mean 100
    b = _pd([200, 210, 220])     # mean 210
    half = blend_price_distributions([a, b])           # default 50/50
    assert half.mean() == pytest.approx(0.5 * 100 + 0.5 * 210)
    tilt = blend_price_distributions([a, b], [0.7, 0.3])
    assert tilt.mean() == pytest.approx(0.7 * 100 + 0.3 * 210)


def test_mixture_renormalises_weights():
    a, b = _pd([100.0]), _pd([200.0])
    # [2, 2] must behave like [0.5, 0.5].
    assert blend_price_distributions([a, b], [2, 2]).mean() == pytest.approx(150.0)


def test_mixture_preserves_component_shape_not_average():
    """A 50/50 blend of two tight clusters keeps BOTH clusters (bimodal), it does not
    collapse to the midpoint — mass sits near 100 and near 200, not at 150."""
    a = _pd(np.full(50, 100.0))
    b = _pd(np.full(50, 200.0))
    out = blend_price_distributions([a, b])
    assert out.quantile(0.25) == pytest.approx(100.0)
    assert out.quantile(0.75) == pytest.approx(200.0)


def test_mixture_rejects_bad_weights():
    a, b = _pd([100.0]), _pd([200.0])
    with pytest.raises(ValueError):
        blend_price_distributions([a, b], [-1, 2])
    with pytest.raises(ValueError):
        blend_price_distributions([a, b], [0, 0])
    with pytest.raises(ValueError):
        blend_price_distributions([])


def test_single_distribution_passthrough():
    a = _pd([90, 100, 110])
    out = blend_price_distributions([a])
    assert out.mean() == pytest.approx(a.mean())


# ---------- BlendForecaster ----------

class _StubForecaster(Forecaster):
    def __init__(self, prices):
        self._pd = _pd(prices)

    def forecast(self, horizon_days, spot):
        return self._pd


def test_blend_forecaster_blends_components():
    f = BlendForecaster([_StubForecaster([90, 100, 110]), _StubForecaster([200, 210, 220])])
    out = f.forecast(21, 100.0)
    assert len(out) == 6
    assert out.mean() == pytest.approx(0.5 * 100 + 0.5 * 210)


def test_blend_forecaster_rejects_bad_weights():
    with pytest.raises(ValueError):
        BlendForecaster([_StubForecaster([100.0]), _StubForecaster([200.0])], [1.0])  # wrong length
    with pytest.raises(ValueError):
        BlendForecaster([])


# ---------- blend backtest plumbing (stubbed index PDF, no network) ----------

def _price_path(returns, p0=100.0):
    return p0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))


def _ts(ticker, returns, p0, start="2024-01-01"):
    closes = _price_path(returns, p0)
    n = len(closes)
    dates = pd.bdate_range(start, periods=n).to_numpy()
    return StockReturnTS(ticker=ticker, dates=dates, open=closes, high=closes,
                         low=closes, close=closes, volume=np.full(n, 1e6),
                         source="synthetic", start=None, end=None)


def _stub_index_pdf(symbol, run_date, horizon):
    s = 400.0 if symbol == "SPY" else 200.0
    k = np.linspace(s * 0.8, s * 1.2, 41)
    return implied_pdf_from_iv(k, np.full_like(k, 0.18), s, horizon / 365.0, 0.04)


def test_blend_backtest_returns_three_aligned_arms():
    rng = np.random.default_rng(0)
    n = 320
    spy = _ts("SPY", rng.normal(0, 0.009, n), 400.0)
    iwm = _ts("IWM", rng.normal(0, 0.011, n), 200.0)
    stock = _ts("XYZ", rng.normal(0.0002, 0.012, n), 100.0)
    cal = EventCalendar("XYZ", [])

    out = rolling_log_score_backtest_blend_event(
        stock, spy, iwm, _stub_index_pdf, cal, horizon_days=21, holdout_days=30,
        n_paths=800, beta_window=None)

    assert set(out) == {"eboot", "oib", "blend"}
    n_eval = 30 - 21 + 1
    assert all(out[a].n == n_eval for a in out)
    # All three arms share the same forecast dates (lockstep).
    dates = [[e.forecast_date for e in out[a].evaluations] for a in out]
    assert dates[0] == dates[1] == dates[2]
    assert all(np.isfinite(out[a].mean_log_score) for a in out)


def test_blend_backtest_skips_when_index_pdf_unavailable():
    rng = np.random.default_rng(1)
    n = 320
    spy = _ts("SPY", rng.normal(0, 0.009, n), 400.0)
    iwm = _ts("IWM", rng.normal(0, 0.011, n), 200.0)
    stock = _ts("XYZ", rng.normal(0, 0.012, n), 100.0)
    cal = EventCalendar("XYZ", [])

    def broken(symbol, run_date, horizon):
        raise ValueError("no chain")

    out = rolling_log_score_backtest_blend_event(
        stock, spy, iwm, broken, cal, horizon_days=21, holdout_days=30, n_paths=500,
        beta_window=None)
    assert all(out[a].n == 0 for a in out)
    assert out["blend"].config["n_skipped"] == 30 - 21 + 1
