"""Tests for VolGatedForecaster — route to FHS vs bootstrap on the vol ratio.

Synthetic, no-network. Verifies the gate endpoints reduce to the two pure
forecasters (τ=0 → FHS, τ=∞ → bootstrap), that the routing decision tracks
σ_{T+1}/σ̄ vs τ, and the uniform-ish constructor.
"""

import numpy as np
import pytest

from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.garch_fhs_forecaster import GarchFhsForecaster
from options_trader.forecast.return_distribution import ReturnDistribution
from options_trader.forecast.vol_gated_forecaster import VolGatedForecaster


def _garch_series(n=1500, omega=2e-6, alpha=0.08, beta=0.90, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    h = omega / (1 - alpha - beta)
    prev = h
    for t in range(n):
        h = omega + alpha * prev + beta * h
        e = np.sqrt(h) * rng.standard_normal()
        r[t] = e
        prev = e ** 2
    return r


SPOT = 100.0


def test_tau_zero_equals_pure_fhs():
    rd = ReturnDistribution(_garch_series())
    gated = VolGatedForecaster(rd, n_paths=4000, seed=9, vol_threshold=0.0)
    fhs = GarchFhsForecaster(rd, n_paths=4000, seed=9)
    assert gated.use_fhs is True
    assert np.array_equal(gated.forecast(10, SPOT).prices, fhs.forecast(10, SPOT).prices)


def test_tau_inf_equals_pure_bootstrap():
    rd = ReturnDistribution(_garch_series())
    gated = VolGatedForecaster(rd, n_paths=4000, seed=9, vol_threshold=float("inf"))
    boot = BootstrapForecaster(rd, n_paths=4000, seed=9)
    assert gated.use_fhs is False
    assert np.array_equal(gated.forecast(10, SPOT).prices, boot.forecast(10, SPOT).prices)


def test_gate_routes_on_vol_ratio():
    """A recent vol spike (ratio > 1) routes to FHS at τ=1.1; a threshold above the
    ratio routes to bootstrap."""
    base = _garch_series(n=1200, seed=1)
    shock = ReturnDistribution(np.concatenate([base, np.tile([0.06, -0.055], 10)]))
    g = VolGatedForecaster(shock, n_paths=100, seed=1, vol_threshold=1.0)
    assert g.vol_ratio > 1.0           # recent turbulence elevates conditional vol
    assert VolGatedForecaster(shock, n_paths=100, seed=1, vol_threshold=1.0).use_fhs is True
    # A threshold just above the realized ratio flips it to bootstrap.
    just_above = g.vol_ratio + 0.5
    assert VolGatedForecaster(shock, n_paths=100, seed=1,
                              vol_threshold=just_above).use_fhs is False


def test_rejects_negative_threshold():
    rd = ReturnDistribution(_garch_series())
    with pytest.raises(ValueError):
        VolGatedForecaster(rd, vol_threshold=-0.1)


def test_uniform_constructor_signature():
    rd = ReturnDistribution(_garch_series())
    g = VolGatedForecaster(rd, 2000, 5, 1.0)
    assert len(g.forecast(5, SPOT)) == 2000
