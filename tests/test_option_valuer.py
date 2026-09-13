"""Tests for OptionValuer.

The cornerstone test (test_ev_matches_black_scholes_under_lognormal) verifies the
conceptual bridge: when the PriceDistribution IS the risk-neutral lognormal of a
BS world, the discounted EV the valuer computes must equal the BS price. That ties
our empirical-EV machinery to a closed form.
"""

import math
from datetime import date

import numpy as np
import pytest

from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.valuation.black_scholes import bs_price
from options_trader.valuation.option_valuer import (
    OptionContract,
    OptionType,
    OptionValuer,
    Recommendation,
)


VAL_DATE = date(2026, 6, 2)


def _contract(strike=100.0, opt=OptionType.CALL, days=30, bid=None, ask=None):
    return OptionContract(
        symbol="TEST",
        underlying="TEST",
        strike=strike,
        expiry=VAL_DATE.fromordinal(VAL_DATE.toordinal() + days),
        option_type=opt,
        bid=bid,
        ask=ask,
    )


def _lognormal_dist(spot, t_years, r, sigma, n=400_000, seed=0):
    """Risk-neutral terminal-price distribution of a BS world."""
    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma * sigma) * t_years
    shock = sigma * math.sqrt(t_years) * rng.standard_normal(n)
    return PriceDistribution(spot * np.exp(drift + shock))


# ---------- the conceptual bridge ----------

def test_ev_matches_black_scholes_under_lognormal():
    spot, r, sigma, days = 100.0, 0.04, 0.25, 30
    t = days / 365.0
    pd = _lognormal_dist(spot, t, r, sigma)
    valuer = OptionValuer(risk_free_rate=r)

    call = valuer.value(pd, _contract(strike=100, days=days), spot, VAL_DATE)
    bs = bs_price(spot, 100, t, r, sigma, "call")
    # Monte Carlo with 400k paths: a couple cents of error is expected.
    assert call.fair_value == pytest.approx(bs, abs=0.05)


def test_put_ev_matches_black_scholes():
    spot, r, sigma, days = 100.0, 0.04, 0.25, 30
    t = days / 365.0
    pd = _lognormal_dist(spot, t, r, sigma)
    valuer = OptionValuer(risk_free_rate=r)
    put = valuer.value(pd, _contract(strike=100, opt=OptionType.PUT, days=days), spot, VAL_DATE)
    assert put.fair_value == pytest.approx(bs_price(spot, 100, t, r, sigma, "put"), abs=0.05)


# ---------- payoff mechanics ----------

def test_expected_payoff_simple_two_point():
    # 50/50 prices 120 and 80, call strike 100 → payoff 20 or 0 → EV 10.
    pd = PriceDistribution(np.array([120.0, 80.0]))
    valuer = OptionValuer(risk_free_rate=0.0)  # no discount
    v = valuer.value(pd, _contract(strike=100), spot=100.0, valuation_date=VAL_DATE)
    assert v.expected_payoff == pytest.approx(10.0)
    assert v.fair_value == pytest.approx(10.0)
    assert v.prob_itm == pytest.approx(0.5)


def test_put_payoff_two_point():
    pd = PriceDistribution(np.array([120.0, 80.0]))
    valuer = OptionValuer(risk_free_rate=0.0)
    v = valuer.value(pd, _contract(strike=100, opt=OptionType.PUT), 100.0, VAL_DATE)
    assert v.expected_payoff == pytest.approx(10.0)
    assert v.prob_itm == pytest.approx(0.5)


def test_weighted_prob_itm():
    pd = PriceDistribution(np.array([120.0, 80.0]), weights=np.array([0.8, 0.2]))
    valuer = OptionValuer(risk_free_rate=0.0)
    v = valuer.value(pd, _contract(strike=100), 100.0, VAL_DATE)
    assert v.prob_itm == pytest.approx(0.8)
    assert v.expected_payoff == pytest.approx(0.8 * 20.0)


def test_discounting_reduces_fair_value():
    pd = PriceDistribution(np.array([120.0, 80.0]))
    undiscounted = OptionValuer(0.0).value(pd, _contract(days=365), 100.0, VAL_DATE)
    discounted = OptionValuer(0.10).value(pd, _contract(days=365), 100.0, VAL_DATE)
    assert discounted.fair_value < undiscounted.fair_value
    assert discounted.fair_value == pytest.approx(math.exp(-0.10 * (365 / 365.0)) * 10.0)


# ---------- edge & recommendation ----------

def test_buy_when_fair_value_exceeds_ask():
    pd = PriceDistribution(np.array([120.0, 80.0]))  # fair value 10
    v = OptionValuer(0.0).value(pd, _contract(strike=100, bid=6.0, ask=7.0), 100.0, VAL_DATE)
    assert v.recommendation == Recommendation.BUY
    assert v.edge_buy == pytest.approx(3.0)
    assert v.edge_pct_buy == pytest.approx(3.0 / 7.0)


def test_sell_when_fair_value_below_bid():
    pd = PriceDistribution(np.array([120.0, 80.0]))  # fair value 10
    v = OptionValuer(0.0).value(pd, _contract(strike=100, bid=12.0, ask=13.0), 100.0, VAL_DATE)
    assert v.recommendation == Recommendation.SELL
    assert v.edge_sell == pytest.approx(2.0)


def test_hold_when_ev_inside_spread():
    pd = PriceDistribution(np.array([120.0, 80.0]))  # fair value 10
    v = OptionValuer(0.0).value(pd, _contract(strike=100, bid=9.0, ask=11.0), 100.0, VAL_DATE)
    assert v.recommendation == Recommendation.HOLD


def test_no_quote_when_missing():
    pd = PriceDistribution(np.array([120.0, 80.0]))
    v = OptionValuer(0.0).value(pd, _contract(strike=100), 100.0, VAL_DATE)
    assert v.recommendation == Recommendation.NO_QUOTE
    assert v.edge_buy is None and v.breakeven is None


def test_breakeven_call_and_put():
    pd = PriceDistribution(np.array([120.0, 80.0]))
    call = OptionValuer(0.0).value(pd, _contract(strike=100, ask=7.0), 100.0, VAL_DATE)
    put = OptionValuer(0.0).value(
        pd, _contract(strike=100, opt=OptionType.PUT, ask=7.0), 100.0, VAL_DATE
    )
    assert call.breakeven == pytest.approx(107.0)
    assert put.breakeven == pytest.approx(93.0)


# ---------- validation ----------

def test_nonpositive_spot_raises():
    pd = PriceDistribution(np.array([120.0, 80.0]))
    with pytest.raises(ValueError, match="spot must be positive"):
        OptionValuer(0.0).value(pd, _contract(), 0.0, VAL_DATE)


def test_expired_contract_raises():
    pd = PriceDistribution(np.array([120.0, 80.0]))
    expired = _contract(days=-1)
    with pytest.raises(ValueError, match="expired"):
        OptionValuer(0.0).value(pd, expired, 100.0, VAL_DATE)


def test_mid_property():
    c = _contract(bid=6.0, ask=8.0)
    assert c.mid == pytest.approx(7.0)
    assert _contract().mid is None


# ---------- edge decomposition (drift vs shape) ----------

def test_decomposition_identity_total_equals_drift_plus_shape():
    spot, r, sigma, days = 100.0, 0.04, 0.30, 30
    t = days / 365.0
    pd = _lognormal_dist(spot, t, r=-0.20, sigma=sigma)  # strong negative drift
    valuer = OptionValuer(risk_free_rate=r)
    contract = _contract(strike=95, opt=OptionType.PUT, days=days, ask=2.0)
    d = valuer.decompose_edge(pd, contract, spot, VAL_DATE)
    # total_edge = drift_edge + shape_edge, exactly
    assert d.total_edge == pytest.approx(d.drift_edge + d.shape_edge, abs=1e-9)
    assert d.drift_share == pytest.approx(d.drift_edge / d.total_edge)


def test_negative_drift_makes_put_edge_drift_positive():
    # A distribution centered well below the forward should attribute positive
    # drift_edge to a put (downward shift raises put EV).
    spot, r, sigma, days = 100.0, 0.0, 0.25, 30
    t = days / 365.0
    pd = _lognormal_dist(spot, t, r=-0.50, sigma=sigma)  # mean far below forward
    valuer = OptionValuer(risk_free_rate=r)
    d = valuer.decompose_edge(pd, _contract(strike=100, opt=OptionType.PUT, days=days, ask=1.0),
                              spot, VAL_DATE)
    assert d.forecast_mean < d.forward
    assert d.drift_edge > 0


def test_no_drift_when_distribution_centered_on_forward():
    # Build a distribution whose mean already equals the forward → drift_edge ≈ 0,
    # so essentially all edge is shape.
    spot, r, sigma, days = 100.0, 0.03, 0.25, 30
    t = days / 365.0
    pd = _lognormal_dist(spot, t, r=r, sigma=sigma)  # risk-neutral: E[S_T] = forward
    valuer = OptionValuer(risk_free_rate=r)
    d = valuer.decompose_edge(pd, _contract(strike=100, days=days, ask=3.0), spot, VAL_DATE)
    assert d.forecast_mean == pytest.approx(d.forward, rel=2e-3)
    assert d.drift_edge == pytest.approx(0.0, abs=0.05)


def test_decomposition_no_quote():
    pd = PriceDistribution(np.array([120.0, 80.0]))
    d = OptionValuer(0.0).decompose_edge(pd, _contract(strike=100), 100.0, VAL_DATE)
    assert d.total_edge is None and d.shape_edge is None and d.drift_share is None
    # drift_edge is still defined (it's EV-based, needs no quote)
    assert isinstance(d.drift_edge, float)


def test_recentering_preserves_shape_coefficient_of_variation():
    # The recenter is multiplicative, so CV (std/mean) is invariant — verify via
    # the fact that valuing an ATM-forward option gives consistent moneyness.
    spot, r, sigma, days = 100.0, 0.05, 0.40, 45
    t = days / 365.0
    pd = _lognormal_dist(spot, t, r=-0.30, sigma=sigma)
    valuer = OptionValuer(risk_free_rate=r)
    forward = valuer.forward_price(spot, VAL_DATE, _contract(days=days).expiry)
    recentered = PriceDistribution(pd.prices * (forward / pd.mean()), pd.weights)
    assert recentered.std() / recentered.mean() == pytest.approx(pd.std() / pd.mean(), rel=1e-9)
    assert recentered.mean() == pytest.approx(forward, rel=1e-9)
