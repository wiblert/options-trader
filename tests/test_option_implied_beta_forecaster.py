"""Tests for forecast/option_implied_beta_forecaster.py — synthetic, no network."""

import logging

import numpy as np
import pytest

from options_trader.forecast.breeden_litzenberger import implied_pdf_from_iv
from options_trader.forecast.option_implied_beta_forecaster import (
    OptionImpliedBetaForecaster,
    idiosyncratic_residuals,
)
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


SPOT = 100.0
T = 21.0 / 365.0
HORIZON = 21


def _index_market(width=0.25, sigma=0.18, spot=400.0):
    k = np.linspace(spot * (1 - width), spot * (1 + width), 41)
    iv = np.full_like(k, sigma)
    return implied_pdf_from_iv(k, iv, spot, T, 0.04).to_return_distribution()


def _idio(scale=0.005, n=400, seed=0):
    return ReturnDistribution(np.random.default_rng(seed).normal(0, scale, n))


def _make(beta_spy=1.0, beta_iwm=1.0, idio_scale=0.005, n_paths=20000, seed=1):
    return OptionImpliedBetaForecaster(
        _index_market(), _index_market(spot=200.0), beta_spy, beta_iwm,
        _idio(idio_scale), n_paths=n_paths, seed=seed, pdf_horizon_days=HORIZON,
    )


# ---------- shape / validity ----------

def test_forecast_shape_and_positive():
    out = _make().forecast(HORIZON, SPOT)
    assert isinstance(out, PriceDistribution)
    assert len(out) == 20000
    assert np.all(out.prices > 0)
    assert np.isfinite(out.mean())


def test_blended_beta_is_average():
    f = _make(beta_spy=0.8, beta_iwm=1.6)
    assert f.blended_beta == pytest.approx(1.2)


# ---------- the systematic mapping: higher beta => wider forecast ----------

def test_higher_beta_widens_forecast():
    lo = _make(beta_spy=0.3, beta_iwm=0.3, idio_scale=1e-4).forecast(HORIZON, SPOT).std()
    hi = _make(beta_spy=2.0, beta_iwm=2.0, idio_scale=1e-4).forecast(HORIZON, SPOT).std()
    assert hi > 2 * lo


def test_zero_beta_leaves_only_idiosyncratic():
    """β=0 => the systematic term vanishes; width comes only from the idio sum."""
    f = _make(beta_spy=0.0, beta_iwm=0.0, idio_scale=0.01)
    out = f.forecast(HORIZON, SPOT)
    # std of sum of HORIZON iid N(0,0.01) daily log-returns, mapped through exp.
    expected_lr_std = 0.01 * np.sqrt(HORIZON)
    realised_lr_std = np.std(np.log(out.prices / SPOT))
    assert realised_lr_std == pytest.approx(expected_lr_std, rel=0.1)


def test_idiosyncratic_adds_width():
    no_idio = _make(beta_spy=1.0, beta_iwm=1.0, idio_scale=1e-5).forecast(HORIZON, SPOT).std()
    big_idio = _make(beta_spy=1.0, beta_iwm=1.0, idio_scale=0.02).forecast(HORIZON, SPOT).std()
    assert big_idio > no_idio


# ---------- reproducibility / validation ----------

def test_reproducible_with_seed():
    a = _make(seed=42).forecast(HORIZON, SPOT)
    b = _make(seed=42).forecast(HORIZON, SPOT)
    assert np.array_equal(a.prices, b.prices)


def test_horizon_mismatch_warns(caplog):
    f = _make()
    with caplog.at_level(logging.WARNING):
        f.forecast(HORIZON + 5, SPOT)
    assert any("horizon" in r.message.lower() for r in caplog.records)


def test_input_validation():
    with pytest.raises(ValueError):
        _make(n_paths=0)
    f = _make()
    with pytest.raises(ValueError):
        f.forecast(0, SPOT)
    with pytest.raises(ValueError):
        f.forecast(HORIZON, -1.0)
    with pytest.raises(ValueError):
        OptionImpliedBetaForecaster(_index_market(), _index_market(), 1.0, 1.0,
                                    _idio(), blend_spy=1.5)


# ---------- idiosyncratic_residuals helper ----------

def test_idiosyncratic_residuals_removes_market_component():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 0.01, 500)
    eps_true = rng.normal(0, 0.004, 500)
    s = 1.3 * m + eps_true
    eps = idiosyncratic_residuals(s, m, 1.3)
    assert np.allclose(eps, eps_true, atol=1e-9)
    # Residual is (nearly) uncorrelated with the market by construction.
    assert abs(np.corrcoef(eps, m)[0, 1]) < 0.05


def test_idiosyncratic_residuals_rejects_mismatch():
    with pytest.raises(ValueError):
        idiosyncratic_residuals(np.zeros(10), np.zeros(8), 1.0)


# ---------- event-conditioned idiosyncratic mode ----------

def test_event_schedule_routes_earnings_to_event_distribution():
    """A horizon day flagged EARNINGS draws from the (downward) earnings residual
    distribution, shifting the forecast mean below an all-normal schedule."""
    from options_trader.data.events.event import EventType

    rng = np.random.default_rng(0)
    normal = ReturnDistribution(rng.normal(0.0, 0.004, 400))
    earnings = ReturnDistribution(rng.normal(-0.06, 0.03, 50))  # clear down-bias + dispersion
    event_dists = {EventType.EARNINGS: earnings}

    def make(schedule):
        return OptionImpliedBetaForecaster(
            _index_market(), _index_market(spot=200.0), 1.0, 1.0, normal,
            n_paths=30000, seed=3, pdf_horizon_days=HORIZON,
            idio_event_dists=event_dists, event_schedule=schedule)

    no_ev = make([None] * HORIZON).forecast(HORIZON, SPOT)
    sched = [None] * HORIZON; sched[10] = EventType.EARNINGS
    with_ev = make(sched).forecast(HORIZON, SPOT)
    # The earnings day injects a ~-6% idiosyncratic draw → lower terminal mean.
    assert with_ev.mean() < no_ev.mean()


def test_event_schedule_length_must_match_horizon():
    from options_trader.data.events.event import EventType
    f = OptionImpliedBetaForecaster(
        _index_market(), _index_market(spot=200.0), 1.0, 1.0, _idio(),
        n_paths=2000, seed=1, idio_event_dists={EventType.EARNINGS: _idio()},
        event_schedule=[None] * (HORIZON + 3))
    with pytest.raises(ValueError):
        f.forecast(HORIZON, SPOT)


def test_plain_mode_unaffected_by_empty_event_args():
    """No event_schedule => bit-identical to the plain vectorised path."""
    a = _make(seed=9).forecast(HORIZON, SPOT)
    b = OptionImpliedBetaForecaster(
        _index_market(), _index_market(spot=200.0), 1.0, 1.0, _idio(0.005),
        n_paths=20000, seed=9, pdf_horizon_days=HORIZON,
        idio_event_dists=None, event_schedule=None).forecast(HORIZON, SPOT)
    assert np.array_equal(a.prices, b.prices)
