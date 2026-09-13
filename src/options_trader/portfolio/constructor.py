"""
constructor.py — portfolio-level allocation across per-ticker candidates.

`run_daily` sizes each ticker independently (fractional Kelly + a per-position
cap). That leaves two cross-ticker gaps this layer closes:

  1. Gross budget — total premium deployed must stay within a fraction of bankroll.
     For LONG options premium = max loss, so the gross-premium cap IS the total
     capital-at-risk cap: "lose at most max_gross_fraction of bankroll if every
     held option expires worthless."
  2. Directional balance — instead of a hard cap on the majority side, we allow a
     one-sided book but PENALISE it: if the funded book is unhedged (one direction
     holds more than hedge_threshold of deployed premium — calls = bullish, puts =
     bearish), trim the overall bet by unhedged_haircut. Rationale: a one-sided
     book carries concentrated market-beta risk, so we down-size it rather than
     refuse the signals outright (the forecaster genuinely sees that edge).

Allocation rule: GREEDY by expected log growth (Kelly's own objective) — fully
fund the best signals top-down until a cap blocks them; skip the rest. We make one
pass at the full budget to see whether the natural book is hedged; if not, we shrink
the gross budget and re-allocate. Existing option premium (open positions) is charged
against the budget so re-runs don't stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from options_trader.valuation.option_valuer import OptionType


_TOL = 1e-9

DIRECTION = {OptionType.CALL: "bullish", OptionType.PUT: "bearish"}


@dataclass(frozen=True)
class PortfolioCaps:
    max_gross_fraction: float = 0.20     # total premium ≤ 20% of bankroll
    hedge_threshold: float = 0.80        # one side > this share of premium = "unhedged"
    unhedged_haircut: float = 0.20       # shrink the overall bet by this when unhedged
    max_per_name_fraction: float = 0.05  # ≤ 5% of bankroll per underlying


@dataclass
class AllocationResult:
    bankroll: float
    gross_budget: float          # effective budget after any unhedged haircut
    current_premium: float       # existing open-option premium charged to budget
    available_budget: float      # gross_budget − current_premium (≥0)
    hedge_haircut_applied: bool = False   # True if the unhedged haircut was triggered
    accepted: list = field(default_factory=list)             # candidates funded
    rejected: list = field(default_factory=list)             # (candidate, reason)

    @property
    def accepted_cost(self) -> float:
        return sum(c.sizing.cost for c in self.accepted)

    @property
    def by_direction(self) -> dict:
        out: dict = {}
        for c in self.accepted:
            d = DIRECTION[c.valuation.contract.option_type]
            out[d] = out.get(d, 0.0) + c.sizing.cost
        return out


class PortfolioConstructor:
    """Greedy, budget-capped allocation with an unhedged-book overall haircut."""

    def __init__(self, caps: Optional[PortfolioCaps] = None) -> None:
        self.caps = caps or PortfolioCaps()

    def allocate(
        self,
        candidates: list,
        bankroll: float,
        current_premium: float = 0.0,
    ) -> AllocationResult:
        """Select which candidates to fund (greedy by log growth).

        Two passes: allocate at the full gross budget; if the resulting book is
        unhedged (one side > hedge_threshold of deployed premium), shrink the gross
        budget by unhedged_haircut and re-allocate. Mutates each candidate's
        `allocated` / `alloc_reason` fields and returns an AllocationResult.
        Candidates are not partially sized — accept or skip.
        """
        if bankroll <= 0:
            raise ValueError(f"bankroll must be positive, got {bankroll}")
        caps = self.caps

        gross = caps.max_gross_fraction * bankroll
        name_cap = caps.max_per_name_fraction * bankroll
        ordered = sorted(candidates, key=lambda c: c.sizing.expected_log_growth, reverse=True)

        # Pass 1: full budget, no directional cap — discover the natural book.
        accepted, rejected = self._greedy(ordered, max(0.0, gross - current_premium), name_cap)

        # If the natural book is one-sided, trim the overall bet and re-allocate.
        haircut_applied = self._is_unhedged(accepted)
        if haircut_applied:
            gross = gross * (1.0 - caps.unhedged_haircut)
            accepted, rejected = self._greedy(ordered, max(0.0, gross - current_premium), name_cap)

        available = max(0.0, gross - current_premium)
        return AllocationResult(
            bankroll=bankroll, gross_budget=gross,
            current_premium=current_premium, available_budget=available,
            hedge_haircut_applied=haircut_applied,
            accepted=accepted, rejected=rejected,
        )

    def _greedy(self, ordered: list, available: float, name_cap: float) -> tuple[list, list]:
        """Greedy accept top-down under the gross + per-name caps. Sets flags."""
        accepted: list = []
        rejected: list = []
        spent = 0.0
        by_name: dict = {}

        for c in ordered:
            cost = c.sizing.cost
            name = c.valuation.contract.underlying

            reason = None
            if spent + cost > available + _TOL:
                reason = "gross_budget"
            elif by_name.get(name, 0.0) + cost > name_cap + _TOL:
                reason = "per_name_cap"

            if reason is not None:
                c.allocated, c.alloc_reason = False, reason
                rejected.append((c, reason))
                continue

            spent += cost
            by_name[name] = by_name.get(name, 0.0) + cost
            c.allocated, c.alloc_reason = True, "accepted"
            accepted.append(c)

        return accepted, rejected

    def _is_unhedged(self, accepted: list) -> bool:
        """True if one direction holds more than hedge_threshold of deployed premium."""
        if not accepted:
            return False
        by_dir: dict = {}
        for c in accepted:
            d = DIRECTION[c.valuation.contract.option_type]
            by_dir[d] = by_dir.get(d, 0.0) + c.sizing.cost
        total = sum(by_dir.values())
        if total <= 0:
            return False
        return max(by_dir.values()) > self.caps.hedge_threshold * total + _TOL
