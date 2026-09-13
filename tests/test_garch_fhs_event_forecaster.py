"""Tests for GarchFhsForecaster event-conditioning mode.

All synthetic / no-network. Key properties verified:
  - Normal-only schedule produces positive, finite prices
  - All-None schedule (explicit) is bit-identical to the plain constructor
  - Event day with high-vol distribution produces a wider terminal distribution
  - GARCH state advances from the event shock: large event move spikes vol for
    subsequent days, making a [event, normal, ...] schedule wider than all-normal
  - Horizon length mismatch raises ValueError
  - Same seed -> same prices (reproducibility)
  - Factory degrades to plain GarchFhsForecaster (no events) when event source is broken
"""

import numpy as np
import pytest

from options_trader.data.events.event import EventType
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions
from options_trader.forecast.garch_fhs_forecaster import GarchFhsForecaster
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


# ── helpers ────────────────────────────────────────────────────────────────────

def _garch_series(n=1500, omega=2e-6, alpha=0.08, beta=0.90, seed=0):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    h = omega / (1 - alpha - beta)
    prev_eps2 = h
    for t in range(n):
        h = omega + alpha * prev_eps2 + beta * h
        eps = np.sqrt(h) * rng.standard_normal()
        r[t] = eps
        prev_eps2 = eps ** 2
    return r


def _small_vol_dist(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return ReturnDistribution(rng.normal(0.0, 0.008, size=n))


def _large_vol_dist(seed=1, n=2000):
    rng = np.random.default_rng(seed)
    return ReturnDistribution(rng.normal(0.0, 0.12, size=n))


def _make_forecaster(event_schedule, *, garch_seed=0, n=1500, n_paths=4000, seed=7):
    full_rd = ReturnDistribution(_garch_series(n=n, seed=garch_seed))
    cond = ConditionedReturnDistributions(
        _small_vol_dist(), {EventType.EARNINGS: _large_vol_dist()}
    )
    return GarchFhsForecaster(
        full_rd, n_paths=n_paths, seed=seed,
        conditioned=cond, event_schedule=event_schedule,
    )


SPOT = 100.0


# ── tests ──────────────────────────────────────────────────────────────────────

def test_positive_finite_prices():
    fc = _make_forecaster([None] * 5)
    pdist = fc.forecast(5, SPOT)
    assert isinstance(pdist, PriceDistribution)
    assert len(pdist) == 4000
    assert np.all(pdist.prices > 0)
    assert np.isfinite(pdist.mean())


def test_normal_only_schedule_is_consistent_with_plain_garch_fhs():
    """GarchFhsForecaster with an all-None event_schedule must produce bit-identical
    prices to the plain constructor (no event params), same seed."""
    full_rd = ReturnDistribution(_garch_series(n=1500, seed=0))
    cond = ConditionedReturnDistributions(_small_vol_dist(), {})

    fc_with_schedule = GarchFhsForecaster(
        full_rd, n_paths=8000, seed=3,
        conditioned=cond, event_schedule=[None] * 10,
    )
    fc_plain = GarchFhsForecaster(full_rd, n_paths=8000, seed=3)

    pe = fc_with_schedule.forecast(10, SPOT)
    pp = fc_plain.forecast(10, SPOT)
    np.testing.assert_array_equal(pe.prices, pp.prices)


def test_event_day_widens_terminal_distribution():
    full_rd = ReturnDistribution(_garch_series(n=1500, seed=0))
    cond = ConditionedReturnDistributions(
        _small_vol_dist(), {EventType.EARNINGS: _large_vol_dist()}
    )

    all_normal = GarchFhsForecaster(
        full_rd, n_paths=30_000, seed=9,
        conditioned=cond, event_schedule=[None] * 5,
    )
    with_earnings = GarchFhsForecaster(
        full_rd, n_paths=30_000, seed=9,
        conditioned=cond,
        event_schedule=[None, None, EventType.EARNINGS, None, None],
    )

    std_normal = all_normal.forecast(5, SPOT).std()
    std_event = with_earnings.forecast(5, SPOT).std()
    assert std_event > 3.0 * std_normal


def test_event_shock_advances_garch_state():
    """A large event shock on day 0 should spike sigma2 and widen subsequent normal days.
    Compare [event, normal*4] vs [normal*5] using a very large-shock event distribution."""
    full_rd = ReturnDistribution(_garch_series(n=1500, seed=0))
    large_shock_dist = ReturnDistribution(
        np.concatenate([np.array([0.20, -0.18] * 500)])  # ±20% shocks
    )
    cond = ConditionedReturnDistributions(_small_vol_dist(), {EventType.EARNINGS: large_shock_dist})

    all_normal = GarchFhsForecaster(
        full_rd, n_paths=30_000, seed=5,
        conditioned=cond, event_schedule=[None] * 5,
    )
    event_first = GarchFhsForecaster(
        full_rd, n_paths=30_000, seed=5,
        conditioned=cond,
        event_schedule=[EventType.EARNINGS, None, None, None, None],
    )

    std_normal = all_normal.forecast(5, SPOT).std()
    std_event_first = event_first.forecast(5, SPOT).std()
    assert std_event_first > 1.5 * std_normal


def test_horizon_mismatch_raises():
    fc = _make_forecaster([None] * 4)
    with pytest.raises(ValueError, match="event_schedule length"):
        fc.forecast(5, SPOT)


def test_reproducibility():
    sched = [None, EventType.EARNINGS, None, None, None]
    full_rd = ReturnDistribution(_garch_series(n=1500, seed=0))
    cond = ConditionedReturnDistributions(
        _small_vol_dist(), {EventType.EARNINGS: _large_vol_dist()}
    )
    a = GarchFhsForecaster(full_rd, n_paths=500, seed=13, conditioned=cond, event_schedule=sched).forecast(5, SPOT)
    b = GarchFhsForecaster(full_rd, n_paths=500, seed=13, conditioned=cond, event_schedule=sched).forecast(5, SPOT)
    np.testing.assert_array_equal(a.prices, b.prices)


def test_input_validation():
    full_rd = ReturnDistribution(_garch_series())
    cond = ConditionedReturnDistributions(_small_vol_dist(), {})
    with pytest.raises(ValueError, match="n_paths"):
        GarchFhsForecaster(full_rd, n_paths=0, conditioned=cond, event_schedule=[None])
    fc = GarchFhsForecaster(full_rd, n_paths=100, seed=1, conditioned=cond, event_schedule=[None] * 5)
    with pytest.raises(ValueError):
        fc.forecast(0, SPOT)
    with pytest.raises(ValueError):
        fc.forecast(5, -1.0)


def test_factory_degrades_to_plain_garch_fhs_on_broken_source():
    """GarchFhsFactory must return a plain GarchFhsForecaster (no events) when the
    event source fails."""
    from datetime import date
    from options_trader.forecast.factories import GarchFhsFactory, ForecastContext
    from options_trader.data.events.source import EventSource

    class BrokenSource(EventSource):
        def get_events(self, ticker, start, end):
            raise RuntimeError("feed down")

    factory = GarchFhsFactory(BrokenSource())
    full_rd = ReturnDistribution(_garch_series())
    import pandas as pd
    dates = pd.date_range("2020-01-02", periods=len(full_rd.samples), freq="B")
    from options_trader.data.stock_return_ts import StockReturnTS
    ts = StockReturnTS(
        ticker="TEST",
        dates=dates.to_numpy(),
        open=np.ones(len(full_rd.samples)) * 100,
        high=np.ones(len(full_rd.samples)) * 101,
        low=np.ones(len(full_rd.samples)) * 99,
        close=np.exp(np.cumsum(full_rd.samples)) * 100,
        volume=np.ones(len(full_rd.samples)) * 1e6,
        source="synthetic",
        start=dates[0].date(),
        end=dates[-1].date(),
    )
    ctx = ForecastContext(
        ticker="TEST", ts=ts, rd=full_rd, horizon=5,
        run_date=date(2024, 6, 1), n_paths=500, seed=7,
    )
    forecaster = factory(ctx)
    assert isinstance(forecaster, GarchFhsForecaster)
    # Plain mode: no event schedule stored
    assert forecaster.event_schedule is None
