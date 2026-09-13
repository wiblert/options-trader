"""Tests for the rolling log-score backtest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from options_trader.backtest.log_score import (
    BacktestResult,
    ForecastEvaluation,
    rolling_log_score_backtest,
)
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.return_distribution import ReturnDistribution


def _synthetic_ts(n: int = 300, mu: float = 0.0005, sigma: float = 0.015,
                  seed: int = 0, ticker: str = "TEST") -> StockReturnTS:
    """Build a synthetic StockReturnTS with geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(mu, sigma, size=n - 1)
    close = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], log_returns])))
    dates = pd.date_range("2024-01-02", periods=n, freq="B").to_numpy()
    return StockReturnTS(
        ticker=ticker,
        dates=dates,
        open=close, high=close, low=close, close=close,
        volume=np.full(n, 1e6),
        source="alpaca",
        start=date(2024, 1, 1),
        end=date(2026, 1, 1),
    )


def _factory(rd: ReturnDistribution) -> BootstrapForecaster:
    return BootstrapForecaster(rd, n_paths=5_000, seed=7)


# ---------- structural ----------

def test_backtest_runs_and_returns_result():
    ts = _synthetic_ts(n=300)
    res = rolling_log_score_backtest(
        ts, _factory, horizon_days=5, holdout_days=50, decay_lambda=0.99,
    )
    assert isinstance(res, BacktestResult)
    assert res.n == 50 - 5 + 1
    assert len(res.evaluations) == res.n


def test_backtest_percentiles_in_unit_interval():
    ts = _synthetic_ts(n=300)
    res = rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=30)
    assert np.all(res.percentiles >= 0.0)
    assert np.all(res.percentiles <= 1.0)


def test_backtest_log_scores_finite():
    ts = _synthetic_ts(n=300)
    res = rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=30)
    assert np.all(np.isfinite(res.log_scores))


def test_backtest_short_ts_raises():
    ts = _synthetic_ts(n=20)
    with pytest.raises(ValueError, match="need at least"):
        rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=63)


def test_backtest_bad_horizon_raises():
    ts = _synthetic_ts(n=300)
    with pytest.raises(ValueError, match="horizon_days"):
        rolling_log_score_backtest(ts, _factory, horizon_days=0, holdout_days=30)


def test_backtest_bad_holdout_raises():
    ts = _synthetic_ts(n=300)
    with pytest.raises(ValueError, match="holdout_days"):
        rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=0)


def test_to_dataframe_shape():
    ts = _synthetic_ts(n=300)
    res = rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=20)
    df = res.to_dataframe()
    assert len(df) == res.n
    assert set(df.columns) >= {
        "forecast_date", "realized_date", "spot", "realized",
        "forecast_mean", "forecast_std", "percentile_of_realized", "log_score",
    }


# ---------- calibration sanity (synthetic GBM should be roughly calibrated) ----------

def test_synthetic_gbm_calibration_is_near_uniform():
    """On a synthetic GBM series, the forecaster should be near-calibrated:
    realised percentiles roughly uniform on [0, 1]."""
    ts = _synthetic_ts(n=600, mu=0.0005, sigma=0.015, seed=0)
    res = rolling_log_score_backtest(
        ts, _factory, horizon_days=5, holdout_days=150, decay_lambda=0.99,
    )
    pcts = res.percentiles

    # KS test against uniform — use a generous threshold (n is small)
    # Empirical CDF should be close to identity
    from scipy.stats import kstest
    stat, pvalue = kstest(pcts, "uniform")
    assert pvalue > 0.01, f"PIT distribution deviates from uniform: p={pvalue}"


# ---------- decay comparison ----------

def test_different_decay_gives_different_log_scores():
    """Different decay values should produce different mean log scores
    (otherwise the parameter is doing nothing)."""
    ts = _synthetic_ts(n=400)
    res_high = rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=50, decay_lambda=0.99)
    res_low  = rolling_log_score_backtest(ts, _factory, horizon_days=5, holdout_days=50, decay_lambda=0.90)
    assert res_high.mean_log_score != pytest.approx(res_low.mean_log_score, abs=0.001)
