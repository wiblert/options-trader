"""Tests for forecast/breeden_litzenberger.py — synthetic, no network.

The ground truth: a FLAT implied-vol smile is a pure Black-Scholes world, so the
recovered risk-neutral PDF must match the analytic lognormal of S_T. We verify
mean ≈ forward, std ≈ analytic lognormal std (at a realistic strike width), that
the density integrates to 1, that a skewed smile produces skewed returns, and the
input guards.
"""

import numpy as np
import pytest

from options_trader.forecast.breeden_litzenberger import (
    ImpliedPDF,
    implied_pdf_from_calls,
    implied_pdf_from_chain,
    implied_pdf_from_iv,
)
from options_trader.valuation.black_scholes import bs_price
from options_trader.valuation.option_valuer import OptionContract, OptionType


SPOT = 500.0
T = 30.0 / 365.0
R = 0.04
SIG = 0.20


def _flat_smile(width=0.25, n=41, sigma=SIG):
    k = np.linspace(SPOT * (1 - width), SPOT * (1 + width), n)
    return k, np.full_like(k, sigma)


def _lognormal_std():
    return SPOT * np.exp(R * T) * np.sqrt(np.exp(SIG ** 2 * T) - 1)


# ---------- flat-vol recovery (the BS ground truth) ----------

def test_flat_vol_pdf_matches_lognormal_moments():
    k, iv = _flat_smile()
    pdf = implied_pdf_from_iv(k, iv, SPOT, T, R)
    assert isinstance(pdf, ImpliedPDF)
    assert pdf.mean_price() == pytest.approx(SPOT * np.exp(R * T), rel=0.01)
    assert pdf.std_price() == pytest.approx(_lognormal_std(), rel=0.05)


def test_density_integrates_to_one():
    k, iv = _flat_smile()
    pdf = implied_pdf_from_iv(k, iv, SPOT, T, R)
    area = np.trapezoid(pdf.density, pdf.strikes)
    assert area == pytest.approx(1.0, abs=1e-6)
    assert np.all(pdf.density >= 0)


def test_from_calls_matches_from_iv():
    k, iv = _flat_smile()
    calls = np.array([bs_price(SPOT, kk, T, R, vv, "call") for kk, vv in zip(k, iv)])
    p_iv = implied_pdf_from_iv(k, iv, SPOT, T, R)
    p_px = implied_pdf_from_calls(k, calls, SPOT, T, R)
    assert p_px.mean_price() == pytest.approx(p_iv.mean_price(), rel=1e-3)
    assert p_px.std_price() == pytest.approx(p_iv.std_price(), rel=1e-3)


# ---------- return distribution bridge ----------

def test_to_return_distribution_weights_normalised():
    k, iv = _flat_smile()
    rd = implied_pdf_from_iv(k, iv, SPOT, T, R).to_return_distribution()
    assert rd.weights.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(rd.samples))
    # Mean log-return is near (r - 0.5σ²)T for a lognormal forward.
    mean_lr = float(np.sum(rd.samples * rd.weights))
    assert mean_lr == pytest.approx((R - 0.5 * SIG ** 2) * T, abs=0.01)


# ---------- skew recovery ----------

def test_put_skew_smile_yields_negative_return_skew():
    # Higher IV at low strikes (equity put skew) => fat left tail in returns.
    k = np.linspace(SPOT * 0.6, SPOT * 1.4, 41)
    iv = 0.30 - 0.0005 * (k - SPOT)
    rd = implied_pdf_from_iv(k, iv, SPOT, T, R).to_return_distribution()
    m = np.sum(rd.samples * rd.weights)
    var = np.sum(rd.weights * (rd.samples - m) ** 2)
    skew = np.sum(rd.weights * (rd.samples - m) ** 3) / var ** 1.5
    assert skew < -0.05


# ---------- chain bridge (duck-typed OptionContract) ----------

def test_from_chain_uses_call_ivs():
    from datetime import date
    k, iv = _flat_smile(n=20)
    contracts = [
        OptionContract(symbol=f"SPY{i}", underlying="SPY", strike=float(kk),
                       expiry=date(2026, 2, 20), option_type=OptionType.CALL,
                       bid=1.0, ask=1.1, implied_vol=float(vv))
        for i, (kk, vv) in enumerate(zip(k, iv))
    ]
    # A put should be ignored.
    contracts.append(OptionContract(symbol="SPYP", underlying="SPY", strike=500.0,
                                    expiry=date(2026, 2, 20), option_type=OptionType.PUT,
                                    implied_vol=0.9))
    pdf = implied_pdf_from_chain(contracts, SPOT, T, R)
    assert pdf.mean_price() == pytest.approx(SPOT * np.exp(R * T), rel=0.01)


# ---------- input validation ----------

def test_rejects_too_few_points():
    with pytest.raises(ValueError):
        implied_pdf_from_iv([490, 500, 510], [0.2, 0.2, 0.2], SPOT, T, R)


def test_rejects_bad_spot_and_expiry():
    k, iv = _flat_smile()
    with pytest.raises(ValueError):
        implied_pdf_from_iv(k, iv, -1.0, T, R)
    with pytest.raises(ValueError):
        implied_pdf_from_iv(k, iv, SPOT, 0.0, R)


def test_chain_too_few_quotes_raises():
    from datetime import date
    contracts = [
        OptionContract(symbol="SPY1", underlying="SPY", strike=500.0,
                       expiry=date(2026, 2, 20), option_type=OptionType.CALL,
                       implied_vol=0.2)
    ]
    with pytest.raises(ValueError):
        implied_pdf_from_chain(contracts, SPOT, T, R)
