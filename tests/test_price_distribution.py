"""Tests for PriceDistribution."""

import numpy as np
import pytest

from options_trader.forecast.price_distribution import PriceDistribution


def test_init_defaults_to_uniform_weights():
    pd = PriceDistribution(np.array([100.0, 110.0, 120.0]))
    np.testing.assert_allclose(pd.weights, [1/3, 1/3, 1/3])


def test_init_normalises_explicit_weights():
    pd = PriceDistribution(np.array([100.0, 110.0]), weights=np.array([3.0, 1.0]))
    np.testing.assert_allclose(pd.weights, [0.75, 0.25])


def test_init_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        PriceDistribution(np.array([]))


def test_init_2d_raises():
    with pytest.raises(ValueError, match="1-D"):
        PriceDistribution(np.zeros((3, 3)))


def test_init_nan_raises():
    with pytest.raises(ValueError, match="NaN/inf"):
        PriceDistribution(np.array([100.0, np.nan, 110.0]))


def test_init_inf_raises():
    with pytest.raises(ValueError, match="NaN/inf"):
        PriceDistribution(np.array([100.0, np.inf, 110.0]))


def test_init_zero_price_raises():
    with pytest.raises(ValueError, match="positive"):
        PriceDistribution(np.array([100.0, 0.0, 110.0]))


def test_init_negative_price_raises():
    with pytest.raises(ValueError, match="positive"):
        PriceDistribution(np.array([100.0, -1.0, 110.0]))


def test_init_weights_length_mismatch_raises():
    with pytest.raises(ValueError, match="weights length"):
        PriceDistribution(np.array([100.0, 110.0]), weights=np.array([0.5, 0.3, 0.2]))


# ---------- moment helpers ----------

def test_mean_uniform():
    pd = PriceDistribution(np.array([100.0, 200.0, 300.0]))
    assert pd.mean() == pytest.approx(200.0)


def test_mean_weighted():
    pd = PriceDistribution(np.array([100.0, 200.0]), weights=np.array([0.1, 0.9]))
    assert pd.mean() == pytest.approx(190.0)


def test_std_uniform():
    pd = PriceDistribution(np.array([100.0, 100.0, 100.0]))
    assert pd.std() == pytest.approx(0.0)


def test_quantile_monotone():
    rng = np.random.default_rng(0)
    prices = rng.lognormal(mean=np.log(100), sigma=0.2, size=10_000)
    pd = PriceDistribution(prices)
    assert pd.quantile(0.05) < pd.quantile(0.5) < pd.quantile(0.95)


def test_quantile_out_of_range_raises():
    pd = PriceDistribution(np.array([100.0, 110.0]))
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        pd.quantile(1.5)
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        pd.quantile(-0.1)


def test_repr_contains_summary():
    pd = PriceDistribution(np.array([100.0, 110.0, 120.0]))
    r = repr(pd)
    assert "n=3" in r
    assert "mean=" in r
    assert "5%=" in r
    assert "95%=" in r
