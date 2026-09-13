"""Tests for forecast/beta.py — synthetic, no network."""

import numpy as np
import pandas as pd
import pytest

from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.beta import (
    BetaCalculator,
    BetaResult,
    align_log_returns,
    compute_beta,
    log_returns,
)


def _ts(ticker, closes, start="2024-01-01"):
    n = len(closes)
    dates = pd.bdate_range(start, periods=n).to_numpy()
    closes = np.asarray(closes, dtype=float)
    z = np.zeros(n)
    return StockReturnTS(
        ticker=ticker, dates=dates,
        open=closes, high=closes, low=closes, close=closes, volume=z + 1.0,
        source="synthetic", start=None, end=None,
    )


def _price_path(returns, p0=100.0):
    return p0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))


# ---------- compute_beta ----------

def test_compute_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, 600)
    s = 1.5 * m + rng.normal(0, 0.003, 600)
    assert compute_beta(s, m, window=None) == pytest.approx(1.5, abs=0.05)


def test_compute_beta_zero_for_independent_series():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, 600)
    s = rng.normal(0, 0.01, 600)
    assert compute_beta(s, m, window=None) == pytest.approx(0.0, abs=0.1)


def test_compute_beta_honours_window():
    # First half beta 0.5, second half beta 2.0; a short window sees only the tail.
    rng = np.random.default_rng(2)
    m = rng.normal(0, 0.01, 600)
    s = np.concatenate([0.5 * m[:300], 2.0 * m[300:]]) + rng.normal(0, 0.001, 600)
    assert compute_beta(s, m, window=250) == pytest.approx(2.0, abs=0.1)


def test_compute_beta_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_beta(np.zeros(50), np.zeros(40), window=None)


def test_compute_beta_rejects_too_few_obs():
    with pytest.raises(ValueError):
        compute_beta(np.zeros(5), np.zeros(5), window=None)


def test_compute_beta_rejects_degenerate_index():
    with pytest.raises(ValueError):
        compute_beta(np.arange(50.0), np.ones(50), window=None)


# ---------- log_returns / align ----------

def test_log_returns_shape_and_dates():
    ts = _ts("X", _price_path(np.full(20, 0.01)))
    dates, r = log_returns(ts)
    assert len(r) == 20 and len(dates) == 20
    assert np.allclose(r, 0.01, atol=1e-9)


def test_align_inner_joins_on_common_dates():
    a = _ts("A", _price_path(np.full(30, 0.01)), start="2024-01-01")
    # B starts 5 business days later -> overlap is shorter.
    b = _ts("B", _price_path(np.full(30, 0.02)), start="2024-01-08")
    dates, (ra, rb) = align_log_returns(a, b)
    assert len(ra) == len(rb) == len(dates)
    assert len(ra) < 30  # the inner join dropped non-overlapping days


# ---------- BetaCalculator ----------

def test_beta_calculator_returns_both_betas():
    rng = np.random.default_rng(3)
    spy_r = rng.normal(0, 0.01, 400)
    iwm_r = rng.normal(0, 0.012, 400)
    stock_r = 1.2 * spy_r + 0.8 * iwm_r + rng.normal(0, 0.002, 400)

    spy = _ts("SPY", _price_path(spy_r))
    iwm = _ts("IWM", _price_path(iwm_r))
    stock = _ts("XYZ", _price_path(stock_r))

    res = BetaCalculator(window=None).compute(stock, spy, iwm)
    assert isinstance(res, BetaResult)
    # Betas are correlated regressions (SPY/IWM not orthogonal) so not exactly the
    # generating coefficients, but both must be clearly positive and finite.
    assert res.beta_spy > 0 and res.beta_iwm > 0
    assert np.isfinite(res.beta_spy) and np.isfinite(res.beta_iwm)
    assert res.n_obs == 400


def test_beta_result_blended():
    res = BetaResult(beta_spy=1.0, beta_iwm=2.0, n_obs=100)
    assert res.blended(0.5) == pytest.approx(1.5)
    assert res.blended(1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        res.blended(1.5)
