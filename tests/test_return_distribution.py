"""Tests for ReturnDistribution."""

import logging

import numpy as np
import pytest

from options_trader.forecast.return_distribution import (
    DEFAULT_DECAY_LAMBDA,
    LARGE_RETURN_THRESHOLD,
    ReturnDistribution,
)


# ---------- construction ----------

def test_init_without_weights_uses_exponential_decay():
    returns = np.array([0.01, 0.02, -0.005, 0.003, -0.001])
    rd = ReturnDistribution(returns)

    # Weights sum to 1
    assert rd.weights.sum() == pytest.approx(1.0)
    # Weights monotonically increase (oldest -> newest)
    assert np.all(np.diff(rd.weights) > 0)
    # Newest weight equals lambda^0 normalised
    raw = DEFAULT_DECAY_LAMBDA ** np.arange(4, -1, -1, dtype=float)
    np.testing.assert_allclose(rd.weights, raw / raw.sum())


def test_init_with_explicit_weights_normalises():
    returns = np.array([0.01, 0.02, -0.005])
    rd = ReturnDistribution(returns, weights=np.array([2.0, 2.0, 6.0]))
    assert rd.weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(rd.weights, [0.2, 0.2, 0.6])


def test_init_empty_returns_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ReturnDistribution(np.array([]))


def test_init_2d_returns_raises():
    with pytest.raises(ValueError, match="1-D"):
        ReturnDistribution(np.zeros((3, 3)))


def test_init_decay_lambda_out_of_range_raises():
    with pytest.raises(ValueError, match="decay_lambda"):
        ReturnDistribution(np.array([0.01, 0.02]), decay_lambda=1.5)
    with pytest.raises(ValueError, match="decay_lambda"):
        ReturnDistribution(np.array([0.01, 0.02]), decay_lambda=0.0)


def test_init_negative_weights_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ReturnDistribution(np.array([0.01, 0.02]), weights=np.array([1.0, -0.5]))


def test_init_zero_sum_weights_raises():
    with pytest.raises(ValueError, match="sum to zero"):
        ReturnDistribution(np.array([0.01, 0.02]), weights=np.array([0.0, 0.0]))


def test_init_weights_length_mismatch_raises():
    with pytest.raises(ValueError, match="weights length"):
        ReturnDistribution(np.array([0.01, 0.02, 0.03]), weights=np.array([0.5, 0.5]))


def test_init_nan_returns_raises():
    with pytest.raises(ValueError, match="NaN/inf"):
        ReturnDistribution(np.array([0.01, np.nan, 0.03]))


def test_init_inf_returns_raises():
    with pytest.raises(ValueError, match="NaN/inf"):
        ReturnDistribution(np.array([0.01, np.inf, 0.03]))


def test_init_large_returns_warn_but_pass(caplog):
    returns = np.array([0.01, 0.6, -0.7, 0.005])  # two large
    with caplog.at_level("WARNING", logger="options_trader.forecast.return_distribution"):
        ReturnDistribution(returns)
    assert any("|log-return|" in r.message for r in caplog.records)


# ---------- from_normal_distribution ----------

def test_from_normal_distribution_uniform_weights():
    rd = ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.01, n_samples=5000, seed=42)
    assert len(rd) == 5000
    np.testing.assert_allclose(rd.weights, np.full(5000, 1 / 5000))


def test_from_normal_distribution_matches_target_moments():
    rd = ReturnDistribution.from_normal_distribution(mean=0.001, variance=0.0004, n_samples=20000, seed=1)
    mean = float(np.sum(rd.samples * rd.weights))
    var = float(np.sum(rd.weights * (rd.samples - mean) ** 2))
    # 20k samples — tolerance ~3 std-err
    assert mean == pytest.approx(0.001, abs=0.001)
    assert var == pytest.approx(0.0004, rel=0.05)


def test_from_normal_distribution_negative_variance_raises():
    with pytest.raises(ValueError, match="variance"):
        ReturnDistribution.from_normal_distribution(mean=0.0, variance=-1.0)


def test_from_normal_distribution_zero_samples_raises():
    with pytest.raises(ValueError, match="n_samples"):
        ReturnDistribution.from_normal_distribution(mean=0.0, variance=0.01, n_samples=0)


# ---------- uniform_weights ----------

def test_uniform_weights_resets():
    rd = ReturnDistribution(np.array([0.01, 0.02, 0.03]))
    rd.uniform_weights()
    np.testing.assert_allclose(rd.weights, [1 / 3, 1 / 3, 1 / 3])


# ---------- smooth_samples ----------

def test_smooth_samples_grows_array_correctly():
    rd = ReturnDistribution(np.array([0.01, 0.02, 0.03]))
    n_before = len(rd)
    rd.smooth_samples(jitter_copies=10, seed=7)
    assert len(rd) == n_before * (1 + 10)


def test_smooth_samples_preserves_weight_sum():
    rd = ReturnDistribution(np.array([0.01, 0.02, 0.03, -0.01, -0.02]))
    rd.smooth_samples(jitter_copies=5, seed=7)
    assert rd.weights.sum() == pytest.approx(1.0)


def test_smooth_samples_default_bandwidth_is_silverman():
    returns = np.random.default_rng(0).normal(0, 0.02, size=500)
    rd = ReturnDistribution(returns, weights=np.ones(500))
    sigma = float(np.std(returns))
    expected_h = 1.06 * sigma * 500 ** (-1 / 5)
    h = rd._silverman_bandwidth()
    assert h == pytest.approx(expected_h, rel=1e-6)


def test_smooth_samples_bad_bandwidth_raises():
    rd = ReturnDistribution(np.array([0.01, 0.02, 0.03]))
    with pytest.raises(ValueError, match="bandwidth"):
        rd.smooth_samples(bandwidth=-0.001)


def test_smooth_samples_bad_jitter_copies_raises():
    rd = ReturnDistribution(np.array([0.01, 0.02, 0.03]))
    with pytest.raises(ValueError, match="jitter_copies"):
        rd.smooth_samples(jitter_copies=0)


def test_smooth_samples_increases_variance_modestly():
    """KDE smoothing inflates variance by approximately bandwidth^2."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.02, size=1000)
    rd = ReturnDistribution(returns, weights=np.ones(1000))
    bw = 0.005
    var_before = float(np.sum(rd.weights * (rd.samples - np.mean(rd.samples)) ** 2))
    rd.smooth_samples(bandwidth=bw, jitter_copies=20, seed=1)
    mean_after = float(np.sum(rd.samples * rd.weights))
    var_after = float(np.sum(rd.weights * (rd.samples - mean_after) ** 2))
    # The expected variance after smoothing increases by ~ k/(k+1) * bw^2
    inflation = var_after - var_before
    expected = (20 / 21) * bw ** 2
    assert inflation == pytest.approx(expected, rel=0.15)


# ---------- repr ----------

def test_repr_contains_summary():
    rd = ReturnDistribution(np.array([0.01, 0.02, -0.005, 0.003]))
    r = repr(rd)
    assert "n=4" in r
    assert "ess=" in r
    assert "mean=" in r
    assert "std=" in r


# ---------- add_sample ----------

def test_add_sample_grows_samples_and_weights():
    rd = ReturnDistribution(np.array([0.01, 0.02]), decay_lambda=0.95)
    rd.add_sample(0.03)
    assert len(rd) == 3
    assert rd.samples[-1] == pytest.approx(0.03)
    assert rd.weights.sum() == pytest.approx(1.0)


def test_add_sample_matches_rebuild_from_scratch():
    """The critical correctness invariant: incremental add equals rebuild."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.02, size=50)
    new_returns = rng.normal(0, 0.02, size=10)
    lam = 0.97

    # Incremental
    rd_inc = ReturnDistribution(base, decay_lambda=lam)
    for r in new_returns:
        rd_inc.add_sample(r)

    # Rebuilt from scratch
    full = np.concatenate([base, new_returns])
    rd_full = ReturnDistribution(full, decay_lambda=lam)

    np.testing.assert_array_equal(rd_inc.samples, rd_full.samples)
    np.testing.assert_allclose(rd_inc.weights, rd_full.weights, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(rd_inc._raw_weights, rd_full._raw_weights, rtol=1e-12)


def test_add_sample_uses_stored_decay_lambda():
    rd = ReturnDistribution(np.array([0.01, 0.02]), decay_lambda=0.5)
    rd.add_sample(0.03)
    # With λ=0.5 and 3 samples, raw weights are [0.25, 0.5, 1.0], normalized [1/7, 2/7, 4/7]
    np.testing.assert_allclose(rd.weights, [1/7, 2/7, 4/7])


def test_add_sample_decay_can_be_overridden_per_call():
    rd = ReturnDistribution(np.array([0.01, 0.02]), decay_lambda=0.99)
    rd.add_sample(0.03, decay_lambda=0.5)
    # Override is one-shot: stored lambda still 0.99 (verify the override took effect)
    # With λ=0.5 on this call: raw = [0.99^1*0.5, 0.99^0*0.5, 1.0] = [0.495, 0.5, 1.0]
    # normalized: [0.495, 0.5, 1.0] / 1.995
    np.testing.assert_allclose(rd.weights, np.array([0.495, 0.5, 1.0]) / 1.995, rtol=1e-12)


def test_add_sample_nonfinite_raises():
    rd = ReturnDistribution(np.array([0.01, 0.02]))
    with pytest.raises(ValueError, match="finite"):
        rd.add_sample(np.nan)
    with pytest.raises(ValueError, match="finite"):
        rd.add_sample(np.inf)


def test_add_sample_invalid_decay_lambda_raises():
    rd = ReturnDistribution(np.array([0.01, 0.02]))
    with pytest.raises(ValueError, match="decay_lambda"):
        rd.add_sample(0.03, decay_lambda=1.5)
    with pytest.raises(ValueError, match="decay_lambda"):
        rd.add_sample(0.03, decay_lambda=0.0)


def test_add_sample_after_uniform_weights():
    """After uniform_weights, raw is all-ones; add_sample with λ=0.5 should
    multiply existing ones by 0.5 and append 1.0, giving normalized [1/4, 1/4, 1/2]."""
    rd = ReturnDistribution(np.array([0.01, 0.02]), decay_lambda=0.5)
    rd.uniform_weights()
    rd.add_sample(0.03)  # uses stored λ=0.5
    np.testing.assert_allclose(rd.weights, [0.25, 0.25, 0.5])


def test_add_sample_preserves_invariants_over_long_sequence():
    """After many add_sample calls, weights still sum to 1 and are non-negative."""
    rd = ReturnDistribution(np.array([0.01]), decay_lambda=0.99)
    rng = np.random.default_rng(1)
    for _ in range(200):
        rd.add_sample(float(rng.normal(0, 0.02)))
    assert len(rd) == 201
    assert rd.weights.sum() == pytest.approx(1.0)
    assert np.all(rd.weights >= 0)
    assert np.all(np.isfinite(rd.weights))


def test_smooth_samples_then_add_sample_normalisation_holds():
    """After smoothing, add_sample should still maintain normalised weights."""
    rd = ReturnDistribution(np.array([0.01, 0.02, -0.01]), decay_lambda=0.95)
    rd.smooth_samples(seed=0, jitter_copies=5)
    n_before = len(rd)
    rd.add_sample(0.005)
    assert len(rd) == n_before + 1
    assert rd.weights.sum() == pytest.approx(1.0)
