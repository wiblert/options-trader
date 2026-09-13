"""
black_scholes.py — analytical Black-Scholes reference prices, greeks, and an
implied-volatility solver.

Why this exists in a project whose edge comes from an empirical PriceDistribution:
Black-Scholes prices an option under the RISK-NEUTRAL (Q) measure assuming
geometric Brownian motion. Our forecasters produce a REAL-WORLD (P) distribution
and the OptionValuer takes expectations under it. Those are different objects.
BS is kept here for two jobs only:

  1. Sanity-check the OptionValuer: feed it a lognormal PriceDistribution whose
     parameters match a BS world and the discounted EV must match bs_price().
  2. Back out the market's implied vol (implied_vol) so we can compare the
     vol the market is charging against the vol our forecast implies.

European exercise, continuous dividend yield q (default 0). Per-share prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm


VALID_TYPES = ("call", "put")


@dataclass(frozen=True)
class Greeks:
    """First-order BS greeks (per share).

    delta : ∂price/∂spot
    gamma : ∂²price/∂spot²
    vega  : ∂price/∂sigma, per 1.00 (=100%) change in vol
    theta : ∂price/∂t, per CALENDAR DAY (annual theta / 365), time decay is negative
    rho   : ∂price/∂r, per 1.00 (=100%) change in the rate
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _validate(option_type: str) -> None:
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be one of {VALID_TYPES}, got {option_type!r}")


def _intrinsic(spot: float, strike: float, option_type: str) -> float:
    return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)


def _d1_d2(spot, strike, t_years, r, sigma, q):
    vol_sqrt_t = sigma * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes European option price (per share).

    Degenerate inputs collapse to discounted intrinsic value:
      * t_years <= 0  → expired, worth its intrinsic value now.
      * sigma   <= 0  → deterministic forward, worth discounted intrinsic.
    """
    _validate(option_type)
    if spot <= 0 or strike <= 0:
        raise ValueError(f"spot and strike must be positive, got spot={spot}, strike={strike}")

    if t_years <= 0:
        return _intrinsic(spot, strike, option_type)
    if sigma <= 0:
        fwd = spot * math.exp((r - q) * t_years)
        return math.exp(-r * t_years) * _intrinsic(fwd, strike, option_type)

    d1, d2 = _d1_d2(spot, strike, t_years, r, sigma, q)
    disc_r = math.exp(-r * t_years)
    disc_q = math.exp(-q * t_years)
    if option_type == "call":
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)


def bs_greeks(
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> Greeks:
    """First-order greeks (per share). See Greeks docstring for conventions."""
    _validate(option_type)
    if spot <= 0 or strike <= 0:
        raise ValueError(f"spot and strike must be positive, got spot={spot}, strike={strike}")
    if t_years <= 0 or sigma <= 0:
        # Greeks are not well-defined at the degenerate boundary; return zeros
        # except a step-function delta for an unambiguous in/out-of-money state.
        intrinsic = _intrinsic(spot, strike, option_type)
        delta = 0.0
        if intrinsic > 0:
            delta = 1.0 if option_type == "call" else -1.0
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(spot, strike, t_years, r, sigma, q)
    disc_r = math.exp(-r * t_years)
    disc_q = math.exp(-q * t_years)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = math.sqrt(t_years)

    gamma = disc_q * pdf_d1 / (spot * sigma * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    common_theta = -(spot * disc_q * pdf_d1 * sigma) / (2 * sqrt_t)
    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta_annual = (
            common_theta
            - r * strike * disc_r * norm.cdf(d2)
            + q * spot * disc_q * norm.cdf(d1)
        )
        rho = strike * t_years * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta_annual = (
            common_theta
            + r * strike * disc_r * norm.cdf(-d2)
            - q * spot * disc_q * norm.cdf(-d1)
        )
        rho = -strike * t_years * disc_r * norm.cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta_annual / 365.0, rho=rho)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    vol_bounds: tuple[float, float] = (1e-4, 5.0),
) -> float:
    """Back out the BS implied volatility that reproduces `price`.

    Brent root-find on bs_price(sigma) - price. Raises ValueError if the price
    is outside the no-arbitrage band (the root is not bracketed), e.g. below
    discounted intrinsic or above the spot.
    """
    _validate(option_type)
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    if t_years <= 0:
        raise ValueError("cannot infer implied vol on an expired option (t_years <= 0)")

    lo, hi = vol_bounds
    f_lo = bs_price(spot, strike, t_years, r, lo, option_type, q) - price
    f_hi = bs_price(spot, strike, t_years, r, hi, option_type, q) - price
    if f_lo * f_hi > 0:
        raise ValueError(
            f"price {price} not bracketed by vols {vol_bounds}; "
            f"likely outside the no-arbitrage band for these terms"
        )
    return float(brentq(
        lambda s: bs_price(spot, strike, t_years, r, s, option_type, q) - price,
        lo, hi, xtol=1e-6, rtol=1e-8,
    ))
