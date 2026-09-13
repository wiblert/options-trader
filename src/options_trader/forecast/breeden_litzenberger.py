"""
breeden_litzenberger.py — extract the risk-neutral PDF from an option smile.

THEORY (Breeden & Litzenberger, 1978)
    The undiscounted risk-neutral density of the underlying's terminal price is
    the SECOND derivative of the call price with respect to strike:

        f_Q(K) = e^{rT} · ∂²C(K) / ∂K²

    Intuition: a tight call spread (long K, short K+dK) pays off only if S_T lands
    in [K, K+dK]; a butterfly (long K-dK, short 2·K, long K+dK) is a discretised
    second difference and pays ≈1 only if S_T ≈ K. Its cost is the discounted
    probability of that bucket, so the second difference of call prices in K, scaled
    up by e^{rT} (undiscounting — the options market bakes the cost of carry / the
    risk-free rate into premiums), recovers the density. (Carr-Madan formalises this.)

    The result is the RISK-NEUTRAL (Q) density, not the real-world (P) one — it is
    the market's forward-looking, fat-tailed, skewed view of terminal index prices.
    For the Option-Implied Beta forecaster that is exactly what we want as the
    systematic-risk baseline; the P-vs-Q distinction (a variance risk premium) is a
    documented V2 refinement.

WHY INDEX OPTIONS
    SPY / IWM are the most liquid option markets in the world, so their smile is
    smooth and dense — a clean PDF. Individual mid-caps have wide spreads and gappy
    strikes that make a direct second derivative pure noise. We extract the PDF from
    the index and map it onto the ticker by Beta (see option_implied_beta_forecaster).

METHOD (this module — pure, no I/O)
    1. Work in implied-vol space, not raw price space. The smile is smooth and
       low-curvature; call prices are steeply convex, so differentiating prices
       directly amplifies quote noise. Fit the IV smile, evaluate on a dense uniform
       grid, and convert each grid IV back to a Black-Scholes call price.
       Two fits (parameter `method`):
         - **"quadratic" (DEFAULT)**: `iv = a + b·m + c·m²` in log-moneyness
           `m = log(K/F)`, optionally volume-weighted. Robust to EOD quote noise —
           it cannot create the negative-`C''` wiggles a free-form spline does. This
           matters: on noisy EOD index smiles the spline's wiggles get clipped to
           zero and artificially NARROW the density (annualised vol came out ~half
           the ATM IV); the quadratic gives annIV ≈ ATM IV and mean = forward.
         - "spline": smoothing cubic spline (good for dense, clean live smiles).
    2. Second derivative numerically on the dense grid (np.gradient twice).
    3. f(K) = e^{rT} · C''(K); clip tiny negatives; normalise to integrate to 1 by
       the trapezoidal rule.

OUTPUT
    `ImpliedPDF` over terminal index prices, with `.to_return_distribution()` to hand
    the forecaster a `ReturnDistribution` of horizon log-returns r = log(K / spot)
    weighted by the density (no decay — these are forward probabilities, not history).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import UnivariateSpline

from options_trader.valuation.black_scholes import bs_price, implied_vol
from options_trader.forecast.return_distribution import ReturnDistribution
from options_trader.config import DEFAULT_RISK_FREE_RATE


logger = logging.getLogger(__name__)
MIN_SMILE_POINTS = 4            # a cubic spline (k=3) needs > k points
DEFAULT_GRID_SIZE = 400         # dense strike grid for the numerical 2nd derivative
DENSITY_FLOOR = 0.0             # negatives from spline wiggle are clipped to this
# Assumed per-strike IV noise (vol points) used to scale the default spline
# smoothing factor s = n · σ². ~0.4 vol-pt is a reasonable index-smile quote noise.
DEFAULT_IV_NOISE = 0.004


@dataclass(frozen=True)
class ImpliedPDF:
    """Risk-neutral density over terminal underlying prices on a dense grid.

    `density` integrates to 1 over `strikes` (trapezoidal). Built by
    `implied_pdf_from_iv` / `implied_pdf_from_calls`.
    """

    strikes: np.ndarray     # dense, sorted, uniform grid of terminal prices K
    density: np.ndarray     # f_Q(K), parallel to strikes, integrates to 1
    spot: float             # underlying spot used to anchor returns
    r: float                # risk-free rate used to undiscount
    t_years: float          # time to expiry (years) the PDF is for

    def mean_price(self) -> float:
        """E_Q[S_T] — should sit near the forward F = spot·e^{rT}."""
        return float(np.trapezoid(self.strikes * self.density, self.strikes))

    def std_price(self) -> float:
        m = self.mean_price()
        var = float(np.trapezoid((self.strikes - m) ** 2 * self.density, self.strikes))
        return float(np.sqrt(max(var, 0.0)))

    def to_return_distribution(self) -> ReturnDistribution:
        """Convert to a ReturnDistribution of horizon log-returns r = log(K/spot).

        The grid cell probabilities (density · dK) become the (non-decayed) sample
        weights — these are forward risk-neutral probabilities, not a time series, so
        the exponential-decay weighting that ReturnDistribution applies to history is
        intentionally bypassed by passing explicit weights.
        """
        log_ret = np.log(self.strikes / self.spot)
        dk = np.gradient(self.strikes)
        weights = self.density * dk
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            raise ValueError("implied PDF has no positive mass; cannot build returns")
        return ReturnDistribution(returns=log_ret, weights=weights)


def _prepare_smile(
    strikes: Sequence[float],
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sort by strike, drop non-finite/non-positive, average duplicate strikes.

    Returns (strikes, values, weights) with weights=None when none supplied.
    Duplicate strikes are averaged (values) / summed (weights).
    """
    k = np.asarray(strikes, dtype=float)
    v = np.asarray(values, dtype=float)
    if k.shape != v.shape or k.ndim != 1:
        raise ValueError(f"strikes/values must be equal-length 1-D, got {k.shape}, {v.shape}")
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != k.shape:
            raise ValueError(f"weights must match strikes, got {w.shape} vs {k.shape}")

    good = np.isfinite(k) & np.isfinite(v) & (k > 0) & (v > 0)
    if w is not None:
        good &= np.isfinite(w)
    k, v = k[good], v[good]
    w = w[good] if w is not None else None

    order = np.argsort(k)
    k, v = k[order], v[order]
    w = w[order] if w is not None else None

    # Collapse duplicate strikes so the fit gets a strictly increasing abscissa:
    # average the IVs, sum the weights (more quotes at a strike => more trust).
    uniq_k, inv = np.unique(k, return_inverse=True)
    if len(uniq_k) != len(k):
        uniq_v = np.zeros_like(uniq_k)
        np.add.at(uniq_v, inv, v)
        uniq_v /= np.bincount(inv)
        if w is not None:
            uniq_w = np.zeros_like(uniq_k)
            np.add.at(uniq_w, inv, w)
            w = uniq_w
        k, v = uniq_k, uniq_v

    return k, v, w


def implied_pdf_from_iv(
    strikes: Sequence[float],
    ivs: Sequence[float],
    spot: float,
    t_years: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    q: float = 0.0,
    *,
    weights: Optional[Sequence[float]] = None,
    method: str = "quadratic",
    smoothing: Optional[float] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> ImpliedPDF:
    """Risk-neutral PDF from an implied-volatility smile (strikes + IVs).

    Args:
        strikes: option strikes (any order; duplicates averaged).
        ivs: Black-Scholes implied vols at each strike (decimals, e.g. 0.20).
        spot: underlying spot price.
        t_years: time to expiry in years (> 0).
        r: continuously-compounded risk-free rate (undiscounting + BS repricing).
        q: continuous dividend yield of the index (default 0).
        weights: optional per-strike weights (e.g. option volume) — only used by the
            "quadratic" fit, where it down-weights stale/illiquid strikes.
        method: smile fit. **"quadratic"** (default) fits `iv = a + b·m + c·m²` in
            log-moneyness `m = log(K/F)`, `F = spot·e^{(r-q)T}` — robust to EOD
            quote noise because it cannot produce the negative-`C''` wiggles a raw
            spline does (those get clipped and artificially narrow the density).
            "spline" keeps the smoothing-cubic-spline path (good for dense, clean
            smiles; over-sensitive to noisy EOD last-trade prices).
        smoothing: spline `s` (method="spline" only). None → heuristic n·noise².
        grid_size: number of points in the dense strike grid.

    Raises:
        ValueError: too few smile points, non-positive spot/expiry, bad method, or a
            degenerate (all-zero-mass) density.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if t_years <= 0:
        raise ValueError(f"t_years must be positive, got {t_years}")
    if method not in ("quadratic", "spline"):
        raise ValueError(f"method must be 'quadratic' or 'spline', got {method!r}")

    k, iv, w = _prepare_smile(strikes, ivs, weights)
    if len(k) < MIN_SMILE_POINTS:
        raise ValueError(f"need >= {MIN_SMILE_POINTS} valid smile points, got {len(k)}")

    # Dense uniform grid spanning the observed strike range.
    grid = np.linspace(k.min(), k.max(), grid_size)

    if method == "quadratic":
        # Volume-weighted quadratic in log-moneyness — smooth, no wiggle artifacts.
        fwd = spot * np.exp((r - q) * t_years)
        m, mg = np.log(k / fwd), np.log(grid / fwd)
        deg = 2 if len(k) >= 3 else 1
        poly_w = np.sqrt(w) if w is not None else None
        coef = np.polyfit(m, iv, deg, w=poly_w)
        iv_grid = np.clip(np.polyval(coef, mg), 1e-4, None)
    else:
        if smoothing is None:
            smoothing = len(k) * (DEFAULT_IV_NOISE ** 2)
        iv_spline = UnivariateSpline(k, iv, k=3, s=smoothing, ext="const")
        iv_grid = np.clip(iv_spline(grid), 1e-4, None)  # vols must stay positive

    # Reprice each grid strike to a call price, then take C''(K) numerically.
    call_grid = np.array([
        bs_price(spot, kk, t_years, r, vv, option_type="call", q=q)
        for kk, vv in zip(grid, iv_grid)
    ])
    d1 = np.gradient(call_grid, grid)
    d2 = np.gradient(d1, grid)

    density = np.exp(r * t_years) * d2
    density = np.clip(density, DENSITY_FLOOR, None)  # spline wiggle can dip < 0

    area = np.trapezoid(density, grid)
    if area <= 0:
        raise ValueError("derived density has non-positive mass; smile likely too noisy")
    density = density / area

    logger.debug(
        "BL PDF: %d smile pts -> grid[%d] (K %.2f-%.2f), spot=%.2f T=%.4f "
        "mean=%.2f fwd=%.2f",
        len(k), grid_size, grid[0], grid[-1], spot, t_years,
        float(np.trapezoid(grid * density, grid)), spot * np.exp((r - q) * t_years),
    )
    return ImpliedPDF(strikes=grid, density=density, spot=spot, r=r, t_years=t_years)


def implied_pdf_from_calls(
    strikes: Sequence[float],
    call_prices: Sequence[float],
    spot: float,
    t_years: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    q: float = 0.0,
    **kwargs,
) -> ImpliedPDF:
    """Risk-neutral PDF from call PRICES (converts each to IV first, then delegates).

    Converting prices to IV before splining is the numerically-stable path: IV is
    smooth and bounded, so the smoothing spline + repricing acts as a no-arbitrage
    interpolation. Strikes whose price is outside the BS no-arb band (root not
    bracketed) are dropped with a warning.
    """
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(call_prices, dtype=float)
    if k.shape != c.shape or k.ndim != 1:
        raise ValueError(f"strikes/call_prices must be equal-length 1-D, got {k.shape}, {c.shape}")

    ks, ivs = [], []
    for kk, cc in zip(k, c):
        if not (np.isfinite(kk) and np.isfinite(cc)) or kk <= 0 or cc <= 0:
            continue
        try:
            ivs.append(implied_vol(cc, spot, kk, t_years, r, option_type="call", q=q))
            ks.append(kk)
        except ValueError:
            logger.debug("Dropping strike %.2f (call %.4f outside no-arb band)", kk, cc)
    if len(ks) < MIN_SMILE_POINTS:
        raise ValueError(
            f"only {len(ks)} call prices yielded a valid IV (need >= {MIN_SMILE_POINTS})"
        )
    return implied_pdf_from_iv(ks, ivs, spot, t_years, r, q, **kwargs)


def implied_pdf_from_chain(
    contracts,
    spot: float,
    t_years: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    q: float = 0.0,
    **kwargs,
) -> ImpliedPDF:
    """Risk-neutral PDF from a list of OptionContract-shaped objects (the live path).

    Duck-typed on `.strike`, `.option_type` (enum with `.value`, or a str), and a
    usable vol/price: prefers `.implied_vol`; otherwise derives IV from the mid
    (or `.ask`/`.bid`). Only CALLS are used — a put-based variant via put-call
    parity is a documented follow-up. In production this is fed live SPY/IWM
    snapshots (component-1 historical bars are bypassed).
    """
    strikes_iv: list[tuple[float, float]] = []
    strikes_px: list[tuple[float, float]] = []
    for c in contracts:
        otype = getattr(c, "option_type", None)
        otype_val = getattr(otype, "value", otype)
        if str(otype_val).lower() != "call":
            continue
        strike = getattr(c, "strike", None)
        if strike is None or strike <= 0:
            continue
        iv = getattr(c, "implied_vol", None)
        if iv is not None and iv > 0:
            strikes_iv.append((float(strike), float(iv)))
            continue
        bid, ask = getattr(c, "bid", None), getattr(c, "ask", None)
        mid = getattr(c, "mid", None)
        px = mid if mid is not None else (
            0.5 * (bid + ask) if bid is not None and ask is not None else None
        )
        if px is not None and px > 0:
            strikes_px.append((float(strike), float(px)))

    # Prefer the IV path when most strikes carry an IV; otherwise fall back to prices.
    if len(strikes_iv) >= MIN_SMILE_POINTS:
        ks, ivs = zip(*strikes_iv)
        return implied_pdf_from_iv(list(ks), list(ivs), spot, t_years, r, q, **kwargs)
    if len(strikes_px) >= MIN_SMILE_POINTS:
        ks, pxs = zip(*strikes_px)
        return implied_pdf_from_calls(list(ks), list(pxs), spot, t_years, r, q, **kwargs)
    raise ValueError(
        f"chain yielded too few usable call quotes "
        f"({len(strikes_iv)} IV, {len(strikes_px)} priced); need >= {MIN_SMILE_POINTS}"
    )
