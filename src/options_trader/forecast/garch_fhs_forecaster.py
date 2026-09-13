"""
garch_fhs_forecaster.py — GARCH(1,1)-filtered Filtered Historical Simulation.

WHY (see docs/research-forecast-accuracy-2026-06.md)
    The plain BootstrapForecaster resamples RAW daily log-returns iid, which
    implicitly assumes constant volatility. Equity vol clusters, so that pool
    mixes calm and turbulent regimes: the forecast is blind to today's vol state
    and cannot let vol evolve over the horizon. FHS fixes both while keeping the
    empirical (fat-tailed) shape that motivated the bootstrap in the first place.

METHOD
    1. FILTER. Fit GARCH(1,1) to the return series and recover the conditional
       ("local") variance h_t = σ_t² for every day via the recursion

           h_t = ω + α·ε_{t-1}² + β·h_{t-1},      ε_t = r_t − μ

       then STANDARDISE each residual by its own local vol:

           z_t = ε_t / σ_t.

       The z_t are ~iid (vol clustering removed) but still fat-tailed (equity
       shocks are leptokurtic beyond clustering), so bootstrapping them iid is
       defensible AND preserves the tails non-parametrically.

    2. SIMULATE. Anchor on TODAY's one-step-ahead variance σ²_{T+1} (from the
       recursion at the end of history). For each Monte-Carlo path, each horizon
       day k:  draw z* uniformly from {z_t};  ε* = σ_k·z*;  r* = μ + ε*;  then
       advance σ²_{k+1} = ω + α·ε*² + β·σ²_k. A large drawn shock raises the next
       day's simulated vol — clustering is reproduced forward, and σ² mean-reverts
       toward ω/(1−α−β). Sum the h daily returns, spot·exp(·) → terminal price.

    3. EVENT DAYS (optional). When `conditioned` and `event_schedule` are supplied,
       horizon days marked with an EventType draw raw returns from the empirical
       event distribution instead of the GARCH-scaled z-draw. The GARCH variance
       still advances from the realised shock magnitude, so a large earnings surprise
       correctly spikes σ² for subsequent normal days. When these params are omitted
       (plain mode), every day uses the GARCH-filtered z-draw — bit-identical to the
       original plain FHS.

DESIGN
    Core constructor: `(distribution, n_paths, seed)` — uniform signature so it
    slots straight into FORECASTER_CLASSES and VolGatedForecaster unchanged.
    Event params are keyword-only optional additions. The GARCH fit uses variance
    targeting (ω pinned to the long-run variance) and Gaussian quasi-MLE over (α, β)
    — robust, 2-parameter, no new dependencies (scipy only).

    Note: the decay weighting carried by the ReturnDistribution is intentionally
    ignored for the GARCH fit — GARCH models the time-variation of vol explicitly,
    which is what an EWMA decay is a crude proxy for, so applying both would
    double-count. (RiskMetrics EWMA is itself a restricted GARCH with ω=0, α+β=1.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from options_trader.data.events.event import EventType
from options_trader.forecast.base import Forecaster
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import ReturnDistribution


logger = logging.getLogger(__name__)


DEFAULT_N_PATHS = 10_000
MIN_SAMPLES = 50            # GARCH needs a reasonable series to filter
MIN_VARIANCE = 1e-12        # below this the series is degenerate
MAX_PERSISTENCE = 0.9995    # cap α+β strictly below 1 (stationarity)
_BIG = 1e12                 # NLL penalty for infeasible parameters


@dataclass(frozen=True)
class GarchFit:
    """Fitted GARCH(1,1) parameters + the filtered series."""

    mu: float
    omega: float
    alpha: float
    beta: float
    h0: float                       # variance-targeting init (= long-run variance)
    cond_var: np.ndarray            # h_t for each historical day (parallel to returns)
    std_resid: np.ndarray           # z_t = ε_t / sqrt(h_t)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def _conditional_variance(
    eps2: np.ndarray, omega: float, beta: float, alpha: float, h0: float
) -> np.ndarray:
    """h_t = ω + α·ε²_{t-1} + β·h_{t-1}, with h_0 = h0 (variance targeting).

    Implemented as a first-order linear recursion via lfilter (C speed):
    h_t = x_t + β·h_{t-1} where x_0 = h0 and x_t = ω + α·ε²_{t-1} for t≥1.
    """
    x = np.empty_like(eps2)
    x[0] = h0
    x[1:] = omega + alpha * eps2[:-1]
    # y[t] = x[t] + beta*y[t-1]  ==  lfilter(b=[1], a=[1, -beta])
    return lfilter([1.0], [1.0, -beta], x)


def fit_garch11(returns: np.ndarray) -> GarchFit:
    """Fit GARCH(1,1) by variance-targeting Gaussian quasi-MLE over (α, β).

    Raises ValueError on too few samples or a degenerate (near-zero) variance.
    Falls back to typical equity params (α=0.05, β=0.90) if the optimiser fails.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < MIN_SAMPLES:
        raise ValueError(f"GARCH needs >= {MIN_SAMPLES} returns, got {len(r)}")

    mu = float(r.mean())
    eps = r - mu
    eps2 = eps ** 2
    target = float(eps2.mean())                 # long-run variance (targeting)
    if target < MIN_VARIANCE:
        raise ValueError(
            f"return variance is degenerate (var={target:.3e} < {MIN_VARIANCE:.0e})"
        )

    def neg_log_lik(params: np.ndarray) -> float:
        alpha, beta = params
        if alpha < 0 or beta < 0 or alpha + beta >= MAX_PERSISTENCE:
            return _BIG
        omega = target * (1.0 - alpha - beta)
        h = _conditional_variance(eps2, omega, beta, alpha, target)
        if not np.all(h > 0) or not np.all(np.isfinite(h)):
            return _BIG
        # Gaussian NLL up to additive constants.
        return float(np.sum(np.log(h) + eps2 / h))

    x0 = np.array([0.05, 0.90])
    bounds = [(0.0, MAX_PERSISTENCE), (0.0, MAX_PERSISTENCE)]
    constraints = ({"type": "ineq", "fun": lambda p: MAX_PERSISTENCE - p[0] - p[1]},)
    try:
        res = minimize(neg_log_lik, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 200, "ftol": 1e-9})
        alpha, beta = (float(res.x[0]), float(res.x[1])) if res.success else (0.05, 0.90)
    except Exception as exc:  # noqa: BLE001 — optimiser edge cases shouldn't abort a ticker
        logger.warning("GARCH MLE failed (%s); using fallback params", exc)
        alpha, beta = 0.05, 0.90

    # Guard the fallback/edge against a boundary persistence.
    if alpha + beta >= MAX_PERSISTENCE:
        scale = MAX_PERSISTENCE / (alpha + beta)
        alpha, beta = alpha * scale * 0.999, beta * scale * 0.999

    omega = target * (1.0 - alpha - beta)
    cond_var = _conditional_variance(eps2, omega, beta, alpha, target)
    std_resid = eps / np.sqrt(cond_var)
    return GarchFit(mu=mu, omega=omega, alpha=alpha, beta=beta, h0=target,
                    cond_var=cond_var, std_resid=std_resid)


class GarchFhsForecaster(Forecaster):
    """GARCH(1,1)-filtered Historical Simulation, with optional event conditioning.

    Plain mode (default): every horizon day draws a GARCH-filtered z and propagates
    the variance recursion forward.

    Event mode (supply `conditioned` + `event_schedule`): horizon days marked with
    an EventType draw raw returns from the empirical event distribution; the GARCH
    state still advances from the realised shock. Normal days use GARCH-filtered draws.
    """

    def __init__(
        self,
        distribution: ReturnDistribution,
        n_paths: int = DEFAULT_N_PATHS,
        seed: Optional[int] = None,
        *,
        conditioned: Optional[ConditionedReturnDistributions] = None,
        event_schedule: Optional[list[Optional[EventType]]] = None,
    ) -> None:
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        # Fit on the raw, time-ordered return series (decay weighting ignored — see
        # module docstring). distribution.samples is oldest -> newest.
        self.fit = fit_garch11(distribution.samples)
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)
        self.conditioned = conditioned
        self.event_schedule = list(event_schedule) if event_schedule is not None else None

        f = self.fit
        # One-step-ahead variance for the FIRST forecast day, from the end of history.
        last_eps2 = float((distribution.samples[-1] - f.mu) ** 2)
        self._sigma2_next = f.omega + f.alpha * last_eps2 + f.beta * float(f.cond_var[-1])

    @property
    def cond_vol_next(self) -> float:
        """Today's one-step-ahead conditional volatility σ_{T+1}."""
        return float(np.sqrt(self._sigma2_next))

    @property
    def uncond_vol(self) -> float:
        """GARCH long-run (unconditional) volatility σ̄ = sqrt(ω/(1−α−β))."""
        return float(np.sqrt(self.fit.h0))

    @property
    def vol_ratio(self) -> float:
        """σ_{T+1} / σ̄ — how far today's conditional vol diverges from baseline.
        >1 = recent spike (bootstrap under-disperses), <1 = recent calm. The gate
        statistic for VolGatedForecaster."""
        return self.cond_vol_next / self.uncond_vol

    def forecast(self, horizon_days: int, spot: float) -> PriceDistribution:
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if self.event_schedule is not None and len(self.event_schedule) != horizon_days:
            raise ValueError(
                f"event_schedule length {len(self.event_schedule)} "
                f"!= horizon_days {horizon_days}"
            )

        f = self.fit
        z_pool = f.std_resid
        sigma2 = np.full(self.n_paths, self._sigma2_next, dtype=float)
        total = np.zeros(self.n_paths, dtype=float)

        for k in range(horizon_days):
            etype = self.event_schedule[k] if self.event_schedule is not None else None
            if etype is not None:
                rd_ev = self.conditioned.distribution_for(etype)
                raw = self.rng.choice(rd_ev.samples, size=self.n_paths, p=rd_ev.weights)
                eps = raw - f.mu
            else:
                z = self.rng.choice(z_pool, size=self.n_paths, replace=True)
                eps = np.sqrt(sigma2) * z
            total += f.mu + eps
            # Advance the conditional variance with the realised (simulated) shock.
            sigma2 = f.omega + f.alpha * eps ** 2 + f.beta * sigma2

        terminal_prices = spot * np.exp(total)
        n_event_days = sum(1 for e in self.event_schedule if e is not None) if self.event_schedule else 0
        logger.debug(
            "GarchFHS: α=%.3f β=%.3f persist=%.3f σ_next=%.4f "
            "horizon=%d event_days=%d spot=%.4f -> mean=%.4f std=%.4f",
            f.alpha, f.beta, f.persistence, np.sqrt(self._sigma2_next),
            horizon_days, n_event_days, spot,
            float(terminal_prices.mean()), float(terminal_prices.std()),
        )
        return PriceDistribution(prices=terminal_prices)
