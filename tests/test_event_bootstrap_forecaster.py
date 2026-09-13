"""Tests for EventConditionedBootstrapForecaster."""

import numpy as np
import pytest

from options_trader.data.events.event import EventType
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions
from options_trader.forecast.event_bootstrap_forecaster import (
    EventConditionedBootstrapForecaster,
)
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


def _normal_dist(seed=0):
    rng = np.random.default_rng(seed)
    return ReturnDistribution(rng.normal(0.0, 0.01, size=2000), weights=np.ones(2000))


def _high_vol_dist(seed=1):
    rng = np.random.default_rng(seed)
    return ReturnDistribution(rng.normal(0.0, 0.10, size=2000), weights=np.ones(2000))


def test_returns_price_distribution():
    cond = ConditionedReturnDistributions(_normal_dist(), {})
    fc = EventConditionedBootstrapForecaster(cond, [None] * 5, n_paths=1000, seed=0)
    pd_ = fc.forecast(horizon_days=5, spot=100.0)
    assert isinstance(pd_, PriceDistribution)
    assert len(pd_) == 1000
    assert np.all(pd_.prices > 0)


def test_schedule_length_mismatch_raises():
    cond = ConditionedReturnDistributions(_normal_dist(), {})
    fc = EventConditionedBootstrapForecaster(cond, [None] * 4, n_paths=100, seed=0)
    with pytest.raises(ValueError, match="event_schedule length"):
        fc.forecast(horizon_days=5, spot=100.0)


def test_all_none_matches_plain_bootstrap_moments():
    dist = _normal_dist(7)
    cond = ConditionedReturnDistributions(dist, {})
    ev = EventConditionedBootstrapForecaster(cond, [None] * 5, n_paths=40_000, seed=3)
    bs = BootstrapForecaster(dist, n_paths=40_000, seed=3)
    pe = ev.forecast(horizon_days=5, spot=100.0)
    pb = bs.forecast(horizon_days=5, spot=100.0)
    # Distributionally equivalent (not bit-identical): means/stds match closely.
    assert pe.mean() == pytest.approx(pb.mean(), rel=0.01)
    assert pe.std() == pytest.approx(pb.std(), rel=0.05)


def test_earnings_day_inflates_terminal_variance():
    cond = ConditionedReturnDistributions(_normal_dist(), {EventType.EARNINGS: _high_vol_dist()})
    seed = 11
    all_normal = EventConditionedBootstrapForecaster(cond, [None] * 5, n_paths=40_000, seed=seed)
    with_earn = EventConditionedBootstrapForecaster(
        cond, [None, None, EventType.EARNINGS, None, None], n_paths=40_000, seed=seed)
    var_normal = all_normal.forecast(5, 100.0).std() ** 2
    var_earn = with_earn.forecast(5, 100.0).std() ** 2
    # The high-vol earnings day should substantially widen the terminal distribution.
    assert var_earn > 3 * var_normal


def test_absent_type_falls_back_to_normal():
    # Schedule references a type not present in by_type -> routes to normal, no error.
    cond = ConditionedReturnDistributions(_normal_dist(), {})
    fc = EventConditionedBootstrapForecaster(
        cond, [EventType.FDA, None, None, None, None], n_paths=2000, seed=0)
    pd_ = fc.forecast(5, 100.0)
    assert len(pd_) == 2000


def test_seed_determinism():
    cond = ConditionedReturnDistributions(_normal_dist(), {EventType.EARNINGS: _high_vol_dist()})
    sched = [None, EventType.EARNINGS, None, None, None]
    a = EventConditionedBootstrapForecaster(cond, sched, n_paths=500, seed=5).forecast(5, 100.0)
    b = EventConditionedBootstrapForecaster(cond, sched, n_paths=500, seed=5).forecast(5, 100.0)
    np.testing.assert_array_equal(a.prices, b.prices)


def test_spot_scales_linearly():
    cond = ConditionedReturnDistributions(_normal_dist(), {})
    a = EventConditionedBootstrapForecaster(cond, [None]*5, n_paths=1000, seed=9).forecast(5, 100.0)
    b = EventConditionedBootstrapForecaster(cond, [None]*5, n_paths=1000, seed=9).forecast(5, 250.0)
    np.testing.assert_allclose(b.prices, 2.5 * a.prices, rtol=1e-10)


def test_n_paths_zero_raises():
    cond = ConditionedReturnDistributions(_normal_dist(), {})
    with pytest.raises(ValueError, match="n_paths"):
        EventConditionedBootstrapForecaster(cond, [None], n_paths=0)
