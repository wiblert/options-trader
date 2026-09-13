"""Tests for BootstrapForecaster and the Forecaster ABC."""

import numpy as np
import pytest

from options_trader.forecast.base import Forecaster
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


# ---------- Forecaster ABC ----------

def test_forecaster_is_abstract():
    with pytest.raises(TypeError):
        Forecaster()  # type: ignore[abstract]


# ---------- construction ----------

def test_bootstrap_forecaster_returns_price_distribution():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=5000, seed=0)
    fc = BootstrapForecaster(rd, n_paths=1000, seed=0)
    pd = fc.forecast(horizon_days=5, spot=100.0)

    assert isinstance(pd, PriceDistribution)
    assert len(pd) == 1000
    assert np.all(pd.prices > 0)


def test_n_paths_zero_raises():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=100, seed=0)
    with pytest.raises(ValueError, match="n_paths"):
        BootstrapForecaster(rd, n_paths=0)


def test_horizon_zero_raises():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=100, seed=0)
    fc = BootstrapForecaster(rd, n_paths=100, seed=0)
    with pytest.raises(ValueError, match="horizon_days"):
        fc.forecast(horizon_days=0, spot=100.0)


def test_negative_spot_raises():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=100, seed=0)
    fc = BootstrapForecaster(rd, n_paths=100, seed=0)
    with pytest.raises(ValueError, match="spot"):
        fc.forecast(horizon_days=5, spot=-1.0)


# ---------- correctness ----------

def test_horizon_one_recovers_lognormal_moments():
    """For r ~ N(mu, var), S_T = S_0 * exp(r) is lognormal.
    Mean of S_T = S_0 * exp(mu + var/2), Var(S_T) = mean^2 * (exp(var) - 1).
    """
    mu = 0.001
    var = 0.0004
    spot = 100.0
    rd = ReturnDistribution.from_normal_distribution(mean=mu, variance=var, n_samples=50_000, seed=1)
    fc = BootstrapForecaster(rd, n_paths=50_000, seed=1)
    pd = fc.forecast(horizon_days=1, spot=spot)

    expected_mean = spot * np.exp(mu + var / 2)
    expected_var = expected_mean ** 2 * (np.exp(var) - 1)

    assert pd.mean() == pytest.approx(expected_mean, rel=0.01)
    assert pd.std() ** 2 == pytest.approx(expected_var, rel=0.05)


def test_horizon_K_variance_scales_with_K():
    """For iid log-returns, K-day variance = K * 1-day variance.
    So Var(log(S_T/S_0)) scales linearly with K.
    """
    mu = 0.0
    var = 0.0004
    spot = 100.0
    rd = ReturnDistribution.from_normal_distribution(mean=mu, variance=var, n_samples=30_000, seed=2)

    # Same seed across both forecasts so RNG state aligns where possible
    fc1 = BootstrapForecaster(rd, n_paths=30_000, seed=42)
    fc5 = BootstrapForecaster(rd, n_paths=30_000, seed=42)
    pd1 = fc1.forecast(horizon_days=1, spot=spot)
    pd5 = fc5.forecast(horizon_days=5, spot=spot)

    var_log_1 = float(np.var(np.log(pd1.prices / spot)))
    var_log_5 = float(np.var(np.log(pd5.prices / spot)))
    # 5-day log-variance should be ~5x 1-day log-variance
    assert var_log_5 / var_log_1 == pytest.approx(5.0, rel=0.10)


def test_spot_scales_linearly():
    """forecast(K, c * spot).prices == c * forecast(K, spot).prices (with same seed)."""
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=5000, seed=3)

    fc_a = BootstrapForecaster(rd, n_paths=1000, seed=99)
    fc_b = BootstrapForecaster(rd, n_paths=1000, seed=99)
    pd_a = fc_a.forecast(horizon_days=5, spot=100.0)
    pd_b = fc_b.forecast(horizon_days=5, spot=250.0)

    np.testing.assert_allclose(pd_b.prices, 2.5 * pd_a.prices, rtol=1e-10)


def test_uniform_path_weights():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=1000, seed=0)
    fc = BootstrapForecaster(rd, n_paths=500, seed=0)
    pd = fc.forecast(horizon_days=3, spot=100.0)
    np.testing.assert_allclose(pd.weights, np.full(500, 1 / 500))


def test_seed_determinism():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=1000, seed=0)
    fc_a = BootstrapForecaster(rd, n_paths=200, seed=7)
    fc_b = BootstrapForecaster(rd, n_paths=200, seed=7)
    pd_a = fc_a.forecast(horizon_days=5, spot=100.0)
    pd_b = fc_b.forecast(horizon_days=5, spot=100.0)
    np.testing.assert_array_equal(pd_a.prices, pd_b.prices)
