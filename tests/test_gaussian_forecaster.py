"""Tests for GaussianForecaster."""

import numpy as np
import pytest

from options_trader.forecast.gaussian_forecaster import GaussianForecaster
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


def test_gaussian_forecaster_returns_price_distribution():
    rd = ReturnDistribution.from_normal_distribution(0.0, 0.0004, n_samples=5000, seed=0)
    fc = GaussianForecaster(rd, n_paths=1000, seed=0)
    pdist = fc.forecast(horizon_days=5, spot=100.0)
    assert isinstance(pdist, PriceDistribution)
    assert len(pdist) == 1000
    assert np.all(pdist.prices > 0)


def test_gaussian_uses_weighted_mean_and_var():
    # Build a distribution with a known weighted mean / variance via explicit weights
    samples = np.array([-0.02, 0.0, 0.02, 0.04])
    weights = np.array([1.0, 1.0, 1.0, 1.0])  # uniform raw
    rd = ReturnDistribution(samples, weights=weights)
    fc = GaussianForecaster(rd, n_paths=100, seed=0)

    expected_mu = float(np.mean(samples))
    expected_var = float(np.mean((samples - expected_mu) ** 2))
    assert fc.mu == pytest.approx(expected_mu)
    assert fc.sigma ** 2 == pytest.approx(expected_var)


def test_gaussian_recovers_lognormal_moments():
    """If μ, σ² are known, the K-day forecast prices should match the
    lognormal mean spot·exp(Kμ + Kσ²/2)."""
    mu = 0.001
    var = 0.0004
    spot = 100.0
    K = 5
    rd = ReturnDistribution.from_normal_distribution(mu, var, n_samples=20_000, seed=1)
    fc = GaussianForecaster(rd, n_paths=50_000, seed=1)
    pdist = fc.forecast(horizon_days=K, spot=spot)

    # Use the empirical sample mean/var of rd (which is what fc actually uses)
    rd_mu = float(np.sum(rd.samples * rd.weights))
    rd_var = float(np.sum(rd.weights * (rd.samples - rd_mu) ** 2))
    expected_mean = spot * np.exp(K * rd_mu + K * rd_var / 2)
    expected_var_price = expected_mean ** 2 * (np.exp(K * rd_var) - 1)

    assert pdist.mean() == pytest.approx(expected_mean, rel=0.02)
    assert pdist.std() ** 2 == pytest.approx(expected_var_price, rel=0.10)


def test_gaussian_horizon_K_variance_scales_with_K():
    mu = 0.0
    var = 0.0004
    spot = 100.0
    rd = ReturnDistribution.from_normal_distribution(mu, var, n_samples=10_000, seed=2)

    fc1 = GaussianForecaster(rd, n_paths=30_000, seed=42)
    fc5 = GaussianForecaster(rd, n_paths=30_000, seed=42)
    pd1 = fc1.forecast(horizon_days=1, spot=spot)
    pd5 = fc5.forecast(horizon_days=5, spot=spot)

    var_log_1 = float(np.var(np.log(pd1.prices / spot)))
    var_log_5 = float(np.var(np.log(pd5.prices / spot)))
    # 5-day log-variance should be ~5× 1-day
    assert var_log_5 / var_log_1 == pytest.approx(5.0, rel=0.05)


def test_gaussian_n_paths_zero_raises():
    rd = ReturnDistribution.from_normal_distribution(0.0, 0.0004, n_samples=100, seed=0)
    with pytest.raises(ValueError, match="n_paths"):
        GaussianForecaster(rd, n_paths=0)


def test_gaussian_zero_variance_raises():
    # A distribution where all samples are identical -> zero variance
    rd = ReturnDistribution(np.array([0.01, 0.01, 0.01]),
                            weights=np.ones(3))
    with pytest.raises(ValueError, match="variance"):
        GaussianForecaster(rd)


def test_gaussian_horizon_zero_raises():
    rd = ReturnDistribution.from_normal_distribution(0.0, 0.0004, n_samples=100, seed=0)
    fc = GaussianForecaster(rd, n_paths=100, seed=0)
    with pytest.raises(ValueError, match="horizon_days"):
        fc.forecast(horizon_days=0, spot=100.0)


def test_gaussian_negative_spot_raises():
    rd = ReturnDistribution.from_normal_distribution(0.0, 0.0004, n_samples=100, seed=0)
    fc = GaussianForecaster(rd, n_paths=100, seed=0)
    with pytest.raises(ValueError, match="spot"):
        fc.forecast(horizon_days=5, spot=-1.0)


def test_gaussian_spot_scales_linearly():
    rd = ReturnDistribution.from_normal_distribution(0.0, 0.0004, n_samples=5000, seed=3)
    fc_a = GaussianForecaster(rd, n_paths=1000, seed=99)
    fc_b = GaussianForecaster(rd, n_paths=1000, seed=99)
    pd_a = fc_a.forecast(horizon_days=5, spot=100.0)
    pd_b = fc_b.forecast(horizon_days=5, spot=250.0)
    np.testing.assert_allclose(pd_b.prices, 2.5 * pd_a.prices, rtol=1e-10)
