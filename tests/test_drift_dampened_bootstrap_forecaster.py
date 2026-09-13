"""Tests for DriftDampenedBootstrapForecaster.

The forecaster draws like the plain bootstrap, then linearly translates every
terminal price so the mean shrinks toward spot by a fraction κ. We verify the
identity at κ=0, the exact mean target at κ=1, the partial shrink in between,
that the translation preserves dispersion/shape, and the input guards.
"""

import numpy as np
import pytest

from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.drift_dampened_bootstrap_forecaster import (
    DriftDampenedBootstrapForecaster,
)
from options_trader.forecast.return_distribution import ReturnDistribution


def _rd(seed: int = 0, n: int = 500, mu: float = 0.002, sigma: float = 0.015):
    # Positive-drift returns so the bootstrap mean sits ABOVE spot — gives the
    # dampening something to pull back.
    rng = np.random.default_rng(seed)
    return ReturnDistribution(rng.normal(mu, sigma, n))


SPOT = 100.0
HORIZON = 10
N_PATHS = 20_000


def test_kappa_zero_is_identical_to_plain_bootstrap():
    rd = _rd()
    plain = BootstrapForecaster(rd, n_paths=N_PATHS, seed=7).forecast(HORIZON, SPOT)
    damp = DriftDampenedBootstrapForecaster(
        rd, n_paths=N_PATHS, seed=7, drift_dampening=0.0
    ).forecast(HORIZON, SPOT)
    # Same seed + zero shift => byte-identical draws.
    assert np.array_equal(plain.prices, damp.prices)


def test_kappa_one_sets_mean_to_spot():
    rd = _rd()
    f = DriftDampenedBootstrapForecaster(rd, n_paths=N_PATHS, seed=7, drift_dampening=1.0)
    pdist = f.forecast(HORIZON, SPOT)
    assert pdist.mean() == pytest.approx(SPOT, abs=1e-9)


def test_partial_dampening_is_linear_in_kappa():
    rd = _rd()
    base_mean = BootstrapForecaster(rd, n_paths=N_PATHS, seed=7).forecast(HORIZON, SPOT).mean()
    for kappa in (0.25, 0.5, 0.75):
        m = DriftDampenedBootstrapForecaster(
            rd, n_paths=N_PATHS, seed=7, drift_dampening=kappa
        ).forecast(HORIZON, SPOT).mean()
        expected = (1 - kappa) * base_mean + kappa * SPOT
        assert m == pytest.approx(expected, abs=1e-9)


def test_translation_preserves_dispersion_and_shape():
    rd = _rd()
    plain = BootstrapForecaster(rd, n_paths=N_PATHS, seed=7).forecast(HORIZON, SPOT)
    damp = DriftDampenedBootstrapForecaster(
        rd, n_paths=N_PATHS, seed=7, drift_dampening=0.6
    ).forecast(HORIZON, SPOT)
    # A pure additive shift leaves std (and all centred moments) unchanged.
    assert damp.std() == pytest.approx(plain.std(), rel=1e-9)
    # Every price moved by the same constant.
    shifts = damp.prices - plain.prices
    assert np.allclose(shifts, shifts[0])


def test_rejects_kappa_out_of_range():
    rd = _rd()
    with pytest.raises(ValueError):
        DriftDampenedBootstrapForecaster(rd, drift_dampening=-0.1)
    with pytest.raises(ValueError):
        DriftDampenedBootstrapForecaster(rd, drift_dampening=1.5)


def test_inherits_bootstrap_input_validation():
    rd = _rd()
    f = DriftDampenedBootstrapForecaster(rd, n_paths=100, seed=1, drift_dampening=0.5)
    with pytest.raises(ValueError):
        f.forecast(0, SPOT)
    with pytest.raises(ValueError):
        f.forecast(HORIZON, -1.0)
