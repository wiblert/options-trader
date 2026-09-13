"""Tests for the Black-Scholes reference module."""

import math

import numpy as np
import pytest

from options_trader.valuation.black_scholes import (
    Greeks,
    bs_greeks,
    bs_price,
    implied_vol,
)


# ---------- price ----------

def test_call_price_known_value():
    # Textbook case: S=100, K=100, T=1, r=5%, sigma=20% → ~10.4506
    price = bs_price(100, 100, 1.0, 0.05, 0.20, "call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_price_known_value():
    # Same params → put ~5.5735
    price = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    # C - P = S - K e^{-rT}
    s, k, t, r, sig = 120, 110, 0.5, 0.03, 0.25
    c = bs_price(s, k, t, r, sig, "call")
    p = bs_price(s, k, t, r, sig, "put")
    assert (c - p) == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)


def test_expired_option_is_intrinsic():
    assert bs_price(120, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(20.0)
    assert bs_price(90, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(0.0)
    assert bs_price(90, 100, -1.0, 0.05, 0.2, "put") == pytest.approx(10.0)


def test_zero_vol_is_discounted_intrinsic():
    # Deterministic forward = S e^{(r)T}; payoff discounted back.
    s, k, t, r = 100, 100, 1.0, 0.05
    fwd = s * math.exp(r * t)
    expected = math.exp(-r * t) * max(fwd - k, 0.0)
    assert bs_price(s, k, t, r, 0.0, "call") == pytest.approx(expected, abs=1e-9)


def test_dividend_yield_lowers_call():
    no_div = bs_price(100, 100, 1.0, 0.05, 0.2, "call", q=0.0)
    with_div = bs_price(100, 100, 1.0, 0.05, 0.2, "call", q=0.03)
    assert with_div < no_div


def test_bad_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        bs_price(100, 100, 1.0, 0.05, 0.2, "straddle")


def test_nonpositive_spot_raises():
    with pytest.raises(ValueError, match="positive"):
        bs_price(0, 100, 1.0, 0.05, 0.2, "call")


# ---------- greeks ----------

def test_call_delta_in_unit_interval():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    assert 0.0 < g.delta < 1.0


def test_put_delta_negative():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "put")
    assert -1.0 < g.delta < 0.0


def test_gamma_vega_positive():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    assert g.gamma > 0 and g.vega > 0


def test_theta_negative_for_long_call():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
    assert g.theta < 0  # time decay


def test_delta_matches_finite_difference():
    s, k, t, r, sig = 100, 100, 0.5, 0.03, 0.25
    g = bs_greeks(s, k, t, r, sig, "call")
    h = 1e-4
    fd = (bs_price(s + h, k, t, r, sig, "call") - bs_price(s - h, k, t, r, sig, "call")) / (2 * h)
    assert g.delta == pytest.approx(fd, abs=1e-5)


def test_vega_matches_finite_difference():
    s, k, t, r, sig = 100, 100, 0.5, 0.03, 0.25
    g = bs_greeks(s, k, t, r, sig, "call")
    h = 1e-5
    fd = (bs_price(s, k, t, r, sig + h, "call") - bs_price(s, k, t, r, sig - h, "call")) / (2 * h)
    assert g.vega == pytest.approx(fd, abs=1e-3)


def test_greeks_degenerate_returns_zeros():
    g = bs_greeks(120, 100, 0.0, 0.05, 0.2, "call")
    assert g == Greeks(delta=1.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)


# ---------- implied vol ----------

def test_implied_vol_round_trips():
    true_sig = 0.32
    price = bs_price(100, 105, 0.5, 0.04, true_sig, "call")
    assert implied_vol(price, 100, 105, 0.5, 0.04, "call") == pytest.approx(true_sig, abs=1e-5)


def test_implied_vol_put_round_trips():
    true_sig = 0.18
    price = bs_price(95, 100, 0.75, 0.02, true_sig, "put")
    assert implied_vol(price, 95, 100, 0.75, 0.02, "put") == pytest.approx(true_sig, abs=1e-5)


def test_implied_vol_out_of_band_raises():
    # Price above spot is impossible for a call → not bracketed.
    with pytest.raises(ValueError, match="bracketed"):
        implied_vol(150, 100, 100, 0.5, 0.05, "call")


def test_implied_vol_expired_raises():
    with pytest.raises(ValueError, match="expired"):
        implied_vol(5.0, 100, 100, 0.0, 0.05, "call")
