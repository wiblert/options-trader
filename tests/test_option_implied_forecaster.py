"""Tests for the Option-Implied (own-options) BENCHMARK forecaster + its backtest.

Synthetic / no-network: a flat-vol smile is turned into a risk-neutral PDF and fed
through the forecaster; the backtest is driven by a stub ticker-PDF function.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from options_trader.forecast.breeden_litzenberger import implied_pdf_from_iv
from options_trader.forecast.option_implied_forecaster import OptionImpliedForecaster
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.backtest.option_implied_backtest import (
    rolling_log_score_backtest_option_implied,
)


SPOT = 100.0
SIGMA = 0.20
R = 0.04
T = 21.0 / 365.0
HORIZON = 21


def _flat_smile_returns(spot=SPOT, sigma=SIGMA, t=T, r=R, width=0.30, n=61):
    """A flat-vol smile -> risk-neutral PDF -> implied ReturnDistribution."""
    k = np.linspace(spot * (1 - width), spot * (1 + width), n)
    iv = np.full_like(k, sigma)
    return implied_pdf_from_iv(k, iv, spot, t, r), \
        implied_pdf_from_iv(k, iv, spot, t, r).to_return_distribution()


def _make(spot=SPOT, **kw):
    _pdf, rd = _flat_smile_returns(spot=spot)
    return OptionImpliedForecaster(rd, pdf_horizon_days=HORIZON, **kw)


# ---------- shape / validity ----------

def test_forecast_shape_and_positive():
    out = _make().forecast(HORIZON, SPOT)
    assert isinstance(out, PriceDistribution)
    assert len(out) > 0
    assert np.all(out.prices > 0)
    assert np.isfinite(out.mean())


def test_forecast_is_deterministic_across_seeds():
    """The PDF grid IS the distribution; seed must not change the output."""
    a = OptionImpliedForecaster(_flat_smile_returns()[1], seed=1).forecast(HORIZON, SPOT)
    b = OptionImpliedForecaster(_flat_smile_returns()[1], seed=999).forecast(HORIZON, SPOT)
    assert a.mean() == pytest.approx(b.mean())
    assert a.std() == pytest.approx(b.std())


# ---------- it reproduces the option-implied PDF ----------

def test_mean_matches_forward_when_anchored_at_pdf_spot():
    """Re-anchored at the PDF's own spot, E[S_T] ~ the forward (flat-vol BL recovers it)."""
    out = _make(spot=SPOT).forecast(HORIZON, SPOT)
    forward = SPOT * np.exp(R * T)
    assert out.mean() == pytest.approx(forward, rel=0.01)


def test_std_matches_implied_pdf():
    pdf, rd = _flat_smile_returns(spot=SPOT)
    out = OptionImpliedForecaster(rd, pdf_horizon_days=HORIZON).forecast(HORIZON, SPOT)
    # Forecast at the PDF's own spot reproduces the PDF's price dispersion.
    assert out.std() == pytest.approx(pdf.std_price(), rel=0.02)


def test_reanchoring_scales_prices_linearly():
    """Forecasting from 2x spot doubles every terminal price (returns are spot-relative)."""
    f = _make(spot=SPOT)
    base = f.forecast(HORIZON, SPOT)
    scaled = f.forecast(HORIZON, 2 * SPOT)
    assert scaled.mean() == pytest.approx(2 * base.mean(), rel=1e-9)
    np.testing.assert_allclose(np.sort(scaled.prices), 2 * np.sort(base.prices), rtol=1e-9)


# ---------- guards / warnings ----------

def test_horizon_mismatch_warns(caplog):
    f = _make()
    with caplog.at_level(logging.WARNING):
        f.forecast(HORIZON + 5, SPOT)
    assert any("does not exist" in r.message or "not rescale" in r.message
               for r in caplog.records)


def test_rejects_bad_spot_and_horizon():
    f = _make()
    with pytest.raises(ValueError):
        f.forecast(HORIZON, 0.0)
    with pytest.raises(ValueError):
        f.forecast(0, SPOT)


def test_rejects_bad_n_paths():
    with pytest.raises(ValueError):
        OptionImpliedForecaster(_flat_smile_returns()[1], n_paths=0)


# ---------- backtest plumbing (stubbed PDF fn, no network) ----------

def _price_path(returns, p0=100.0):
    return p0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))


def _ts(ticker, returns, p0, start="2024-01-01"):
    closes = _price_path(returns, p0)
    n = len(closes)
    dates = pd.bdate_range(start, periods=n).to_numpy()
    return StockReturnTS(ticker=ticker, dates=dates, open=closes, high=closes,
                         low=closes, close=closes, volume=np.full(n, 1e6),
                         source="synthetic", start=None, end=None)


def _stub_pdf(symbol, run_date, horizon):
    s = 100.0
    k = np.linspace(s * 0.7, s * 1.3, 61)
    return implied_pdf_from_iv(k, np.full_like(k, SIGMA), s, horizon / 365.0, R)


def test_backtest_runs_with_stub_pdf():
    rng = np.random.default_rng(0)
    stock = _ts("XYZ", rng.normal(0, 0.012, 320), 100.0)
    res = rolling_log_score_backtest_option_implied(
        stock, _stub_pdf, horizon_days=21, holdout_days=30, n_paths=2000)
    assert res.n == 30 - 21 + 1
    assert res.config["n_skipped"] == 0
    assert np.isfinite(res.mean_log_score)


def test_backtest_skips_when_pdf_unavailable():
    rng = np.random.default_rng(1)
    stock = _ts("XYZ", rng.normal(0, 0.012, 320), 100.0)

    def broken_pdf(symbol, run_date, horizon):
        raise ValueError("no chain")

    res = rolling_log_score_backtest_option_implied(
        stock, broken_pdf, horizon_days=21, holdout_days=30, n_paths=2000)
    assert res.n == 0
    assert res.config["n_skipped"] == 30 - 21 + 1
