"""
kelly.py — fractional-Kelly position sizing for a long option.

The Kelly criterion sizes a bet to maximize the expected log growth of bankroll.
For a long option held to expiry the per-dollar return is a random variable over
our forecast PriceDistribution:

    R_i = payoff_i / premium - 1            payoff_i = max(S_i - K, 0)  (call)

where premium is what we pay to open (the ask). R_i ∈ [-1, ∞): the worst case is
losing the whole premium (option expires worthless), so the Kelly domain is
f ∈ [0, 1). The optimal full-Kelly fraction f* maximizes

    g(f) = Σ w_i · log(1 + f · R_i)

We then deliberately deploy LESS than f*:
  * fractional Kelly (default ¼) — f* assumes the distribution is exactly right;
    it isn't (see docs/issues.md ISSUE-1: the forecast mean is a noisy drift), so
    we shrink to cut the cost of estimation error and reduce drawdowns.
  * a hard max-fraction CAP — a backstop so one over-confident signal cannot
    deploy an outsized share of bankroll (the ISSUE-1 "48 correlated bets are
    really one bet" guard; also the natural place for the 100x contract multiplier).

V1: long positions only (buying calls/puts). Sizing short/written options (which
carry large/unbounded loss and a different Kelly domain) is future work. Single
position at a time — no cross-position correlation accounting yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.valuation.option_valuer import OptionType, OptionValuation


DEFAULT_KELLY_FRACTION = 0.25   # ¼-Kelly
DEFAULT_MAX_FRACTION = 0.20     # never deploy >20% of bankroll on one position
DEFAULT_CONTRACT_MULTIPLIER = 100


class SizingStatus(str, Enum):
    SIZED = "SIZED"                          # n_contracts >= 1
    NO_EDGE = "NO_EDGE"                       # E[R] <= 0 → Kelly says don't bet
    NO_QUOTE = "NO_QUOTE"                     # no ask to buy at
    BELOW_ONE_CONTRACT = "BELOW_ONE_CONTRACT" # edge exists but bankroll·fraction < 1 contract


def optimal_kelly_fraction(returns: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Full-Kelly fraction f* ∈ [0, 1) maximizing E[log(1 + f·R)].

    Assumes a bounded-loss long bet: every return R >= -1 (worst case loses the
    whole stake), so 1 + f·R > 0 on the whole domain f ∈ [0, 1).

    Returns 0.0 when the weighted-mean return is <= 0 (no favorable bet).
    """
    R = np.asarray(returns, dtype=float)
    if R.ndim != 1 or R.size == 0:
        raise ValueError("returns must be a non-empty 1-D array")
    if np.any(R < -1.0):
        raise ValueError("returns must be >= -1 (a long bet cannot lose more than the stake)")

    if weights is None:
        w = np.ones(R.size) / R.size
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != R.shape:
            raise ValueError("weights must match returns shape")
        w = w / w.sum()

    if float(np.sum(w * R)) <= 0.0:      # g'(0) = E[R]; no edge ⇒ don't bet
        return 0.0

    hi = 1.0 - 1e-9                       # boundary of the no-ruin domain
    g_prime = lambda f: float(np.sum(w * R / (1.0 + f * R)))
    if g_prime(hi) >= 0.0:                # optimum sits at/above the boundary
        return hi
    return float(brentq(g_prime, 0.0, hi, xtol=1e-10))


def expected_log_growth(returns: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    """g(fraction) = Σ w·log(1 + fraction·R). Per-period expected log growth."""
    R = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(np.sum(w * np.log1p(fraction * R)))


@dataclass(frozen=True)
class KellySizing:
    """Result of sizing one long option position."""

    valuation: OptionValuation
    bankroll: float
    full_kelly_fraction: float    # f* (unscaled)
    target_fraction: float        # kelly_fraction · f* (before the cap)
    applied_fraction: float       # after the max-fraction cap
    deployed_fraction: float      # actual = cost / bankroll, after integer rounding
    n_contracts: int
    cost: float                   # n_contracts · premium · multiplier
    premium_per_contract: float   # ask · multiplier
    expected_return: float        # E[R] per dollar of premium
    expected_log_growth: float    # g(deployed_fraction)
    capped: bool                  # the max-fraction cap bound
    status: SizingStatus

    def __repr__(self) -> str:
        return (
            f"KellySizing({self.valuation.contract.symbol} {self.status.value} "
            f"n={self.n_contracts} cost={self.cost:.0f} "
            f"f*={self.full_kelly_fraction:.3f} applied={self.applied_fraction:.3f})"
        )


class KellySizer:
    """Sizes a long option position via fractional Kelly with a hard cap."""

    def __init__(
        self,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        max_fraction: float = DEFAULT_MAX_FRACTION,
        contract_multiplier: int = DEFAULT_CONTRACT_MULTIPLIER,
    ) -> None:
        if not (0.0 < kelly_fraction <= 1.0):
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        if not (0.0 < max_fraction <= 1.0):
            raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}")
        if contract_multiplier < 1:
            raise ValueError(f"contract_multiplier must be >= 1, got {contract_multiplier}")
        self.kelly_fraction = float(kelly_fraction)
        self.max_fraction = float(max_fraction)
        self.contract_multiplier = int(contract_multiplier)

    def size(
        self,
        price_dist: PriceDistribution,
        valuation: OptionValuation,
        bankroll: float,
    ) -> KellySizing:
        """Size a long BUY of `valuation.contract` against `price_dist`.

        Args:
            price_dist: the SAME terminal-price distribution the valuation used
                (held-to-expiry invariant). Drives the per-dollar return law.
            valuation: the OptionValuer output; `buy_price` (ask) is the premium.
            bankroll: total account equity in dollars.

        Returns:
            KellySizing. n_contracts == 0 with a NO_* / NO_EDGE status when no
            position should be opened.
        """
        if bankroll <= 0:
            raise ValueError(f"bankroll must be positive, got {bankroll}")

        premium = valuation.buy_price  # the ask, per share
        contract = valuation.contract
        strike = contract.strike

        def _empty(status: SizingStatus, premium_per_contract: float = 0.0) -> KellySizing:
            return KellySizing(
                valuation=valuation, bankroll=bankroll,
                full_kelly_fraction=0.0, target_fraction=0.0, applied_fraction=0.0,
                deployed_fraction=0.0, n_contracts=0, cost=0.0,
                premium_per_contract=premium_per_contract,
                expected_return=0.0, expected_log_growth=0.0,
                capped=False, status=status,
            )

        if premium is None or premium <= 0:
            return _empty(SizingStatus.NO_QUOTE)

        # Per-dollar return of buying this option, over the forecast paths.
        prices = price_dist.prices
        if contract.option_type == OptionType.CALL:
            payoff = np.maximum(prices - strike, 0.0)
        else:
            payoff = np.maximum(strike - prices, 0.0)
        returns = payoff / premium - 1.0
        weights = price_dist.weights

        premium_per_contract = premium * self.contract_multiplier
        expected_return = float(np.sum((weights / weights.sum()) * returns))

        f_star = optimal_kelly_fraction(returns, weights)
        if f_star <= 0.0:
            out = _empty(SizingStatus.NO_EDGE, premium_per_contract)
            return KellySizing(**{**out.__dict__, "expected_return": expected_return})

        target = self.kelly_fraction * f_star
        applied = min(target, self.max_fraction)
        capped = applied < target

        n_contracts = int(np.floor((applied * bankroll) / premium_per_contract))
        if n_contracts < 1:
            out = _empty(SizingStatus.BELOW_ONE_CONTRACT, premium_per_contract)
            return KellySizing(**{
                **out.__dict__,
                "full_kelly_fraction": f_star, "target_fraction": target,
                "applied_fraction": applied, "expected_return": expected_return,
                "capped": capped,
            })

        cost = n_contracts * premium_per_contract
        deployed_fraction = cost / bankroll
        g = expected_log_growth(returns, weights, deployed_fraction)

        return KellySizing(
            valuation=valuation, bankroll=bankroll,
            full_kelly_fraction=f_star, target_fraction=target, applied_fraction=applied,
            deployed_fraction=deployed_fraction, n_contracts=n_contracts, cost=cost,
            premium_per_contract=premium_per_contract,
            expected_return=expected_return, expected_log_growth=g,
            capped=capped, status=SizingStatus.SIZED,
        )
