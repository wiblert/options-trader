"""Tests for fractional-Kelly option sizing.

The math core (optimal_kelly_fraction) is checked against the closed-form binary
Kelly f* = p - q/b. The sizer is checked end-to-end through the real OptionValuer
using a 2-point PriceDistribution, which makes a call a clean binary bet.
"""

from datetime import date

import numpy as np
import pytest

from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.sizing.kelly import (
    KellySizer,
    SizingStatus,
    expected_log_growth,
    optimal_kelly_fraction,
)
from options_trader.valuation.option_valuer import (
    OptionContract,
    OptionType,
    OptionValuer,
)


VAL_DATE = date(2026, 6, 2)


def _call(strike, ask, days=30):
    return OptionContract(
        symbol="TEST", underlying="TEST", strike=strike,
        expiry=VAL_DATE.fromordinal(VAL_DATE.toordinal() + days),
        option_type=OptionType.CALL, bid=ask * 0.98, ask=ask,
    )


# ---------- optimal_kelly_fraction: closed-form binary Kelly ----------

def test_even_money_positive_edge():
    # win +1 (p=0.6) / lose -1 (q=0.4), b=1 → f* = p - q/b = 0.2
    f = optimal_kelly_fraction(np.array([1.0, -1.0]), np.array([0.6, 0.4]))
    assert f == pytest.approx(0.2, abs=1e-6)


def test_b_two_positive_edge():
    # b=2, p=0.6 → f* = 0.6 - 0.4/2 = 0.4
    f = optimal_kelly_fraction(np.array([2.0, -1.0]), np.array([0.6, 0.4]))
    assert f == pytest.approx(0.4, abs=1e-6)


def test_fair_bet_returns_zero():
    f = optimal_kelly_fraction(np.array([1.0, -1.0]), np.array([0.5, 0.5]))
    assert f == 0.0


def test_negative_edge_returns_zero():
    f = optimal_kelly_fraction(np.array([1.0, -1.0]), np.array([0.4, 0.6]))
    assert f == 0.0


def test_derivative_is_zero_at_optimum():
    R = np.array([3.0, -1.0])
    w = np.array([0.4, 0.6])
    f = optimal_kelly_fraction(R, w)
    g_prime = float(np.sum((w / w.sum()) * R / (1.0 + f * R)))
    assert g_prime == pytest.approx(0.0, abs=1e-6)


def test_uniform_weights_default():
    # [+1, -1] uniform → fair bet → 0
    assert optimal_kelly_fraction(np.array([1.0, -1.0])) == 0.0


def test_returns_below_minus_one_raises():
    with pytest.raises(ValueError, match=">= -1"):
        optimal_kelly_fraction(np.array([1.0, -1.5]))


def test_empty_returns_raises():
    with pytest.raises(ValueError, match="non-empty"):
        optimal_kelly_fraction(np.array([]))


# ---------- expected_log_growth ----------

def test_expected_log_growth_zero_fraction_is_zero():
    assert expected_log_growth(np.array([1.0, -1.0]), np.array([0.6, 0.4]), 0.0) == 0.0


def test_expected_log_growth_matches_manual():
    R = np.array([1.0, -1.0]); w = np.array([0.6, 0.4]); f = 0.2
    manual = 0.6 * np.log(1.2) + 0.4 * np.log(0.8)
    assert expected_log_growth(R, w, f) == pytest.approx(manual)


# ---------- KellySizer end-to-end (binary via 2-point PriceDistribution) ----------

def _binary_setup(p_up=0.6, s_up=120.0, s_dn=80.0, strike=100.0, ask=10.0):
    pd = PriceDistribution(np.array([s_up, s_dn]), weights=np.array([p_up, 1 - p_up]))
    valuer = OptionValuer(risk_free_rate=0.0)
    val = valuer.value(pd, _call(strike, ask), spot=100.0, valuation_date=VAL_DATE)
    return pd, val


def test_sizer_sizes_binary_edge():
    # payoff [20, 0], ask 10 → R=[1,-1], p=0.6 → f*=0.2; ¼-Kelly→applied≈0.05.
    # bankroll 130k keeps the floor off the integer boundary: 0.05*130k/1000=6.5 → 6.
    pd, val = _binary_setup()
    sizer = KellySizer(kelly_fraction=0.25, max_fraction=0.50)
    s = sizer.size(pd, val, bankroll=130_000)
    assert s.status == SizingStatus.SIZED
    assert s.full_kelly_fraction == pytest.approx(0.2, abs=1e-6)
    assert s.applied_fraction == pytest.approx(0.05, abs=1e-6)
    assert not s.capped
    # the sizing invariant holds exactly, independent of brentq epsilon
    assert s.n_contracts == int(np.floor(s.applied_fraction * 130_000 / s.premium_per_contract))
    assert s.n_contracts == 6
    assert s.cost == pytest.approx(6000.0)
    assert s.deployed_fraction == pytest.approx(s.cost / 130_000)


def test_sizer_no_edge_returns_zero():
    pd, val = _binary_setup(p_up=0.5)  # fair → no edge
    s = KellySizer().size(pd, val, bankroll=100_000)
    assert s.status == SizingStatus.NO_EDGE
    assert s.n_contracts == 0 and s.cost == 0.0
    assert s.expected_return == pytest.approx(0.0, abs=1e-9)


def test_sizer_cap_binds():
    # f*=0.2, kelly_fraction=1.0 → target 0.2, cap 0.05 → applied 0.05, capped
    pd, val = _binary_setup()
    sizer = KellySizer(kelly_fraction=1.0, max_fraction=0.05)
    s = sizer.size(pd, val, bankroll=100_000)
    assert s.capped
    assert s.applied_fraction == pytest.approx(0.05)
    assert s.target_fraction == pytest.approx(0.2)


def test_sizer_below_one_contract():
    # tiny bankroll: 0.05 * 500 = $25 < $1000 per contract
    pd, val = _binary_setup()
    s = KellySizer(kelly_fraction=0.25, max_fraction=0.5).size(pd, val, bankroll=500)
    assert s.status == SizingStatus.BELOW_ONE_CONTRACT
    assert s.n_contracts == 0
    assert s.full_kelly_fraction == pytest.approx(0.2, abs=1e-6)  # edge still reported


def test_sizer_no_quote():
    pd = PriceDistribution(np.array([120.0, 80.0]), weights=np.array([0.6, 0.4]))
    contract = OptionContract(
        symbol="TEST", underlying="TEST", strike=100.0,
        expiry=VAL_DATE.fromordinal(VAL_DATE.toordinal() + 30),
        option_type=OptionType.CALL,  # no bid/ask
    )
    val = OptionValuer(0.0).value(pd, contract, spot=100.0, valuation_date=VAL_DATE)
    s = KellySizer().size(pd, val, bankroll=100_000)
    assert s.status == SizingStatus.NO_QUOTE
    assert s.n_contracts == 0


def test_integer_rounding_floors():
    # ask 11 → R=[20/11-1, -1]=[0.818,-1], p=0.6 → f* = 0.6 - 0.4/0.818 = 0.111.
    # applied = 0.25*0.111 = 0.0278 → 0.0278*100k = 2778; /1100 = 2.53 → floors to 2.
    pd, val = _binary_setup(ask=11.0)
    sizer = KellySizer(kelly_fraction=0.25, max_fraction=0.5)
    s = sizer.size(pd, val, bankroll=100_000)
    assert s.full_kelly_fraction == pytest.approx(0.6 - 0.4 / (20 / 11 - 1), abs=1e-6)
    assert s.n_contracts == 2
    assert s.cost == pytest.approx(2200.0)
    assert s.deployed_fraction == pytest.approx(0.022)


def test_bankroll_must_be_positive():
    pd, val = _binary_setup()
    with pytest.raises(ValueError, match="bankroll must be positive"):
        KellySizer().size(pd, val, bankroll=0)


def test_invalid_sizer_params():
    with pytest.raises(ValueError, match="kelly_fraction"):
        KellySizer(kelly_fraction=0.0)
    with pytest.raises(ValueError, match="max_fraction"):
        KellySizer(max_fraction=1.5)
    with pytest.raises(ValueError, match="contract_multiplier"):
        KellySizer(contract_multiplier=0)
