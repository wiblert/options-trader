"""Tests for BlockBootstrapForecaster (moving-block K-day bootstrap)."""

import numpy as np
import pytest

from options_trader.forecast.base import Forecaster
from options_trader.forecast.block_bootstrap_forecaster import BlockBootstrapForecaster
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


# ---------- construction / interface ----------

def test_is_a_forecaster():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=500, seed=0)
    assert isinstance(BlockBootstrapForecaster(rd), Forecaster)


def test_returns_price_distribution():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=500, seed=0)
    fc = BlockBootstrapForecaster(rd, n_paths=1000, seed=0)
    pd = fc.forecast(horizon_days=5, spot=100.0)
    assert isinstance(pd, PriceDistribution)
    assert len(pd) == 1000
    assert np.all(pd.prices > 0)


def test_n_paths_zero_raises():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=100, seed=0)
    with pytest.raises(ValueError, match="n_paths"):
        BlockBootstrapForecaster(rd, n_paths=0)


def test_horizon_zero_raises():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=100, seed=0)
    fc = BlockBootstrapForecaster(rd, n_paths=100, seed=0)
    with pytest.raises(ValueError, match="horizon_days"):
        fc.forecast(horizon_days=0, spot=100.0)


def test_negative_spot_raises():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=100, seed=0)
    fc = BlockBootstrapForecaster(rd, n_paths=100, seed=0)
    with pytest.raises(ValueError, match="spot"):
        fc.forecast(horizon_days=5, spot=-1.0)


def test_horizon_longer_than_samples_raises():
    rd = ReturnDistribution(np.array([0.01, -0.02, 0.005]))  # only 3 daily samples
    fc = BlockBootstrapForecaster(rd, n_paths=50, seed=0)
    with pytest.raises(ValueError, match="at least horizon_days"):
        fc.forecast(horizon_days=5, spot=100.0)


def test_horizon_equal_to_n_yields_single_block():
    """With n samples and K=n there is exactly one overlapping block: the whole
    series. Every path must produce the same terminal price = spot*exp(sum)."""
    returns = np.array([0.01, -0.02, 0.03, 0.005])
    rd = ReturnDistribution(returns)
    fc = BlockBootstrapForecaster(rd, n_paths=200, seed=0)
    pd = fc.forecast(horizon_days=len(returns), spot=100.0)
    expected = 100.0 * np.exp(returns.sum())
    np.testing.assert_allclose(pd.prices, expected, rtol=1e-12)


# ---------- correctness ----------

def test_blocks_are_contiguous_sums():
    """Every drawn K-day log-return must equal some contiguous K-day window sum
    of the underlying series — never a sum of non-adjacent days."""
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.02, size=40)
    rd = ReturnDistribution(returns, weights=np.ones(40))  # uniform so all blocks reachable
    K = 5
    fc = BlockBootstrapForecaster(rd, n_paths=5000, seed=1)
    pd = fc.forecast(horizon_days=K, spot=100.0)

    drawn_log_returns = np.log(pd.prices / 100.0)
    valid_blocks = np.array([returns[j:j + K].sum() for j in range(len(returns) - K + 1)])
    # Each drawn value must match one of the valid contiguous block sums.
    for v in np.unique(drawn_log_returns):
        assert np.any(np.isclose(v, valid_blocks, atol=1e-9)), v


def test_spot_scales_linearly():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=500, seed=3)
    fc_a = BlockBootstrapForecaster(rd, n_paths=1000, seed=99)
    fc_b = BlockBootstrapForecaster(rd, n_paths=1000, seed=99)
    pd_a = fc_a.forecast(horizon_days=5, spot=100.0)
    pd_b = fc_b.forecast(horizon_days=5, spot=250.0)
    np.testing.assert_allclose(pd_b.prices, 2.5 * pd_a.prices, rtol=1e-10)


def test_uniform_path_weights():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=500, seed=0)
    fc = BlockBootstrapForecaster(rd, n_paths=400, seed=0)
    pd = fc.forecast(horizon_days=3, spot=100.0)
    np.testing.assert_allclose(pd.weights, np.full(400, 1 / 400))


def test_seed_determinism():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.0004, n_samples=500, seed=0)
    fc_a = BlockBootstrapForecaster(rd, n_paths=200, seed=7)
    fc_b = BlockBootstrapForecaster(rd, n_paths=200, seed=7)
    pd_a = fc_a.forecast(horizon_days=5, spot=100.0)
    pd_b = fc_b.forecast(horizon_days=5, spot=100.0)
    np.testing.assert_array_equal(pd_a.prices, pd_b.prices)


def test_end_anchored_weighting_favors_recent_blocks():
    """With strong decay, the most recent block (ending at the newest day) should
    dominate sampling, so the forecast concentrates near the last K-day window."""
    # Two regimes: old returns ~ +0.0, recent returns ~ +0.05/day.
    returns = np.concatenate([np.zeros(50), np.full(5, 0.05)])
    rd = ReturnDistribution(returns, decay_lambda=0.5)  # aggressive recency
    fc = BlockBootstrapForecaster(rd, n_paths=20_000, seed=0)
    pd = fc.forecast(horizon_days=5, spot=100.0)
    # The newest block sums to 5*0.05 = 0.25 -> price ~ 100*exp(0.25) ~ 128.4.
    recent_block_price = 100.0 * np.exp(0.25)
    # Heavy recency weighting should pull the forecast mean well above spot.
    assert pd.mean() > 110.0
    # And the most-recent block should be the modal outcome.
    assert np.isclose(pd.quantile(0.95), recent_block_price, rtol=0.02)


def test_block_preserves_serial_dependence_iid_does_not():
    """On a perfectly trending series (every day +0.01), a 5-day block always
    sums to 0.05. The iid bootstrap, drawing 5 independent days WITH replacement,
    produces the same 0.05 only because all days are identical here — so to make
    the distinction visible we use a series with alternating signs that cancels
    over contiguous windows but NOT under independent resampling."""
    # Alternating +a,-a,...: any contiguous even-length window sums to ~0,
    # but independent draws can pick several +a in a row (nonzero sums).
    a = 0.03
    returns = np.tile([a, -a], 25)  # length 50, contiguous 2/4/...-day sums ~ 0
    rd_block = ReturnDistribution(returns, weights=np.ones(50))
    rd_iid = ReturnDistribution(returns, weights=np.ones(50))

    block = BlockBootstrapForecaster(rd_block, n_paths=20_000, seed=0)
    iid = BootstrapForecaster(rd_iid, n_paths=20_000, seed=0)

    pd_block = block.forecast(horizon_days=4, spot=100.0)
    pd_iid = iid.forecast(horizon_days=4, spot=100.0)

    block_log = np.log(pd_block.prices / 100.0)
    iid_log = np.log(pd_iid.prices / 100.0)

    # Contiguous 4-day windows of an alternating series all sum to 0 -> zero spread.
    assert np.std(block_log) < 1e-9
    # Independent resampling lets +a runs survive -> strictly positive spread.
    assert np.std(iid_log) > 0.5 * a
