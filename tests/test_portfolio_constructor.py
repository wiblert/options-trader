"""Tests for the portfolio constructor (greedy allocation under caps). Pure logic."""

from types import SimpleNamespace

import pytest

from options_trader.portfolio.constructor import PortfolioCaps, PortfolioConstructor
from options_trader.valuation.option_valuer import OptionType


def _cand(cost, growth, otype, name):
    """A duck-typed Candidate exposing the fields the constructor reads."""
    contract = SimpleNamespace(option_type=otype, underlying=name)
    return SimpleNamespace(
        sizing=SimpleNamespace(cost=cost, expected_log_growth=growth),
        valuation=SimpleNamespace(contract=contract),
        allocated=False, alloc_reason=None,
    )


# ---------- gross budget ----------

def test_gross_budget_binds_and_drops_worst():
    # bankroll 100k → gross 20k. Four 8k candidates, distinct names, mixed dirs.
    caps = PortfolioCaps(max_gross_fraction=0.20, hedge_threshold=1.0, max_per_name_fraction=1.0)
    cands = [
        _cand(8000, 4.0, OptionType.CALL, "A"),
        _cand(8000, 3.0, OptionType.PUT, "B"),
        _cand(8000, 2.0, OptionType.CALL, "C"),
        _cand(8000, 1.0, OptionType.PUT, "D"),
    ]
    res = PortfolioConstructor(caps).allocate(cands, bankroll=100_000)
    # 8k + 8k = 16k fits; third would be 24k > 20k
    assert [c.valuation.contract.underlying for c in res.accepted] == ["A", "B"]
    assert all(r == "gross_budget" for _, r in res.rejected)
    assert res.accepted_cost == 16_000


def test_greedy_prefers_highest_log_growth():
    caps = PortfolioCaps(max_gross_fraction=0.10, hedge_threshold=1.0, max_per_name_fraction=1.0)
    cands = [
        _cand(5000, 1.0, OptionType.CALL, "LOW"),
        _cand(5000, 9.0, OptionType.CALL, "HIGH"),
    ]  # only one 5k fits in a 10k... actually both fit (10k); test ordering with a tighter budget
    caps = PortfolioCaps(max_gross_fraction=0.05, hedge_threshold=1.0, max_per_name_fraction=1.0)
    res = PortfolioConstructor(caps).allocate(cands, bankroll=100_000)
    assert len(res.accepted) == 1
    assert res.accepted[0].valuation.contract.underlying == "HIGH"


# ---------- unhedged-book haircut ----------

def test_unhedged_book_triggers_haircut():
    # gross 20k. Four all-bullish 5k calls → pass 1 funds all (20k, 100% one side
    # > 80% threshold) → unhedged → budget cut 20% to 16k → re-allocate funds 3 (15k).
    caps = PortfolioCaps(max_gross_fraction=0.20, hedge_threshold=0.80,
                         unhedged_haircut=0.20, max_per_name_fraction=1.0)
    cands = [
        _cand(5000, 5.0, OptionType.CALL, "A"),
        _cand(5000, 4.0, OptionType.CALL, "B"),
        _cand(5000, 3.0, OptionType.CALL, "C"),
        _cand(5000, 2.0, OptionType.CALL, "D"),
    ]
    res = PortfolioConstructor(caps).allocate(cands, bankroll=100_000)
    assert res.hedge_haircut_applied is True
    assert res.gross_budget == pytest.approx(16_000)
    assert [c.valuation.contract.underlying for c in res.accepted] == ["A", "B", "C"]
    assert res.accepted_cost == 15_000
    assert ("D", "gross_budget") in [(c.valuation.contract.underlying, r) for c, r in res.rejected]


def test_hedged_book_no_haircut():
    # 10k bullish + 5k bearish = 66.7% majority < 80% → hedged → full 20k budget kept.
    caps = PortfolioCaps(max_gross_fraction=0.20, hedge_threshold=0.80,
                         unhedged_haircut=0.20, max_per_name_fraction=1.0)
    cands = [
        _cand(10_000, 5.0, OptionType.CALL, "A"),
        _cand(5_000, 4.0, OptionType.PUT, "B"),
    ]
    res = PortfolioConstructor(caps).allocate(cands, bankroll=100_000)
    assert res.hedge_haircut_applied is False
    assert res.gross_budget == pytest.approx(20_000)
    assert res.accepted_cost == 15_000
    assert res.by_direction == {"bullish": 10_000, "bearish": 5_000}


def test_haircut_flag_set_even_when_budget_does_not_rebind():
    # 9k bullish + 1k bearish = 90% > 80% → unhedged. Budget cut to 16k but the
    # 10k book still fits → both stay funded; only the flag/budget reflect the cut.
    caps = PortfolioCaps(max_gross_fraction=0.20, hedge_threshold=0.80,
                         unhedged_haircut=0.20, max_per_name_fraction=1.0)
    cands = [
        _cand(9_000, 5.0, OptionType.CALL, "A"),
        _cand(1_000, 4.0, OptionType.PUT, "B"),
    ]
    res = PortfolioConstructor(caps).allocate(cands, bankroll=100_000)
    assert res.hedge_haircut_applied is True
    assert res.gross_budget == pytest.approx(16_000)
    assert res.accepted_cost == 10_000


# ---------- per-name cap ----------

def test_per_name_cap():
    caps = PortfolioCaps(max_gross_fraction=1.0, hedge_threshold=1.0, max_per_name_fraction=0.05)
    cands = [
        _cand(3000, 5.0, OptionType.CALL, "SAME"),
        _cand(3000, 4.0, OptionType.PUT, "SAME"),  # 3k+3k = 6k > 5k cap
    ]
    res = PortfolioConstructor(caps).allocate(cands, bankroll=100_000)
    assert len(res.accepted) == 1
    assert res.rejected[0][1] == "per_name_cap"


# ---------- existing positions charged to budget ----------

def test_current_premium_reduces_budget():
    caps = PortfolioCaps(max_gross_fraction=0.20, hedge_threshold=1.0, max_per_name_fraction=1.0)
    cands = [_cand(8000, 5.0, OptionType.CALL, "A")]
    # gross 20k, already 15k deployed → only 5k available → 8k candidate rejected
    res = PortfolioConstructor(caps).allocate(cands, bankroll=100_000, current_premium=15_000)
    assert res.available_budget == 5_000
    assert not res.accepted and res.rejected[0][1] == "gross_budget"


# ---------- bookkeeping ----------

def test_sets_allocated_flags():
    caps = PortfolioCaps(max_gross_fraction=0.20, hedge_threshold=1.0, max_per_name_fraction=1.0)
    cands = [_cand(8000, 5.0, OptionType.CALL, "A"), _cand(20000, 1.0, OptionType.PUT, "B")]
    PortfolioConstructor(caps).allocate(cands, bankroll=100_000)
    assert cands[0].allocated is True and cands[0].alloc_reason == "accepted"
    assert cands[1].allocated is False and cands[1].alloc_reason == "gross_budget"


def test_empty_candidates():
    res = PortfolioConstructor().allocate([], bankroll=100_000)
    assert res.accepted == [] and res.rejected == []


def test_bankroll_must_be_positive():
    with pytest.raises(ValueError, match="bankroll must be positive"):
        PortfolioConstructor().allocate([], bankroll=0)
