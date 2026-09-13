"""Tests for the GARCH(1,1)-filtered FHS forecaster.

Synthetic, no-network. We generate a series WITH genuine volatility clustering
(simulate a GARCH path) so the filter has structure to recover, and verify:
the fit is stationary and standardises the residuals to ~unit variance; the
forecast is shape-valid; the forecast width is CONDITIONAL on the recent vol
state (the core FHS property); reproducibility; and input validation. Also
checks the uniform (distribution, n_paths, seed) constructor so it drops into
FORECASTER_CLASSES.
"""

import numpy as np
import pytest

from options_trader.forecast.garch_fhs_forecaster import (
    GarchFhsForecaster,
    fit_garch11,
)
from options_trader.forecast.return_distribution import ReturnDistribution


def _garch_series(n=1500, omega=2e-6, alpha=0.08, beta=0.90, seed=0):
    """Simulate a GARCH(1,1) return path (oldest -> newest)."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    h = omega / (1 - alpha - beta)      # start at long-run variance
    prev_eps2 = h
    for t in range(n):
        h = omega + alpha * prev_eps2 + beta * h
        eps = np.sqrt(h) * rng.standard_normal()
        r[t] = eps
        prev_eps2 = eps ** 2
    return r


SPOT = 100.0


def test_fit_is_stationary_and_standardises_residuals():
    r = _garch_series()
    fit = fit_garch11(r)
    assert 0.0 <= fit.alpha
    assert 0.0 <= fit.beta
    assert fit.persistence < 1.0            # stationary
    assert len(fit.cond_var) == len(r)
    assert np.all(fit.cond_var > 0)
    # Filtering should remove the heteroskedasticity: z_t ~ unit variance.
    assert fit.std_resid.std() == pytest.approx(1.0, abs=0.15)
    # And recover meaningful persistence from a strongly-clustered series.
    assert fit.persistence > 0.8


def test_forecast_shape_and_positive():
    rd = ReturnDistribution(_garch_series())
    f = GarchFhsForecaster(rd, n_paths=5000, seed=7)
    pdist = f.forecast(horizon_days=10, spot=SPOT)
    assert len(pdist) == 5000
    assert np.all(pdist.prices > 0)
    assert np.isfinite(pdist.mean())


def test_forecast_width_is_conditional_on_recent_vol():
    """The defining FHS property: a turbulent recent tape => wider forecast than
    an otherwise-identical history ending calm."""
    base = _garch_series(n=1200, seed=1)
    calm_tail = np.full(20, 0.0005)                       # tiny moves
    shock_tail = np.tile([0.06, -0.055], 10)              # violent moves

    calm = ReturnDistribution(np.concatenate([base, calm_tail]))
    shock = ReturnDistribution(np.concatenate([base, shock_tail]))

    std_calm = GarchFhsForecaster(calm, n_paths=8000, seed=3).forecast(5, SPOT).std()
    std_shock = GarchFhsForecaster(shock, n_paths=8000, seed=3).forecast(5, SPOT).std()

    # Recent turbulence must widen the forecast materially.
    assert std_shock > 1.3 * std_calm


def test_reproducible_with_seed():
    rd = ReturnDistribution(_garch_series())
    a = GarchFhsForecaster(rd, n_paths=3000, seed=11).forecast(7, SPOT)
    b = GarchFhsForecaster(rd, n_paths=3000, seed=11).forecast(7, SPOT)
    assert np.array_equal(a.prices, b.prices)


def test_uniform_constructor_signature_for_registry():
    """Must be buildable as cls(rd, n_paths, seed) to slot into FORECASTER_CLASSES."""
    rd = ReturnDistribution(_garch_series())
    f = GarchFhsForecaster(rd, 2000, 5)
    assert len(f.forecast(5, SPOT)) == 2000


def test_input_validation():
    rd = ReturnDistribution(_garch_series())
    with pytest.raises(ValueError):
        GarchFhsForecaster(rd, n_paths=0)
    f = GarchFhsForecaster(rd, n_paths=100, seed=1)
    with pytest.raises(ValueError):
        f.forecast(0, SPOT)
    with pytest.raises(ValueError):
        f.forecast(5, -1.0)


def test_rejects_too_few_or_degenerate():
    with pytest.raises(ValueError):
        fit_garch11(np.zeros(10))                 # too few
    with pytest.raises(ValueError):
        fit_garch11(np.zeros(100))                # degenerate (zero variance)
