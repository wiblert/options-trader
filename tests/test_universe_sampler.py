"""Tests for the cap-weighted universe sampler.

All tests use a synthetic in-memory snapshot DataFrame — no network, no files.
"""

import numpy as np
import pandas as pd
import pytest

from options_trader.universe.sampler import (
    DUAL_CLASS_ISSUERS,
    _dedup_dual_class,
    sample_tickers,
    sampling_weights,
)


def _snapshot(caps: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": list(caps), "market_cap": list(caps.values()),
         "snapshot_date": "2026-06-02"}
    )


@pytest.fixture
def snap():
    # One mega-cap dwarfing a long tail of equal small-caps.
    caps = {"BIG": 1_000_000.0}
    caps.update({f"S{i}": 1_000.0 for i in range(50)})
    return _snapshot(caps)


# ---------- determinism / reproducibility ----------

def test_same_seed_same_draw(snap):
    a = sample_tickers(10, seed=42, snapshot=snap)
    b = sample_tickers(10, seed=42, snapshot=snap)
    assert a == b


def test_different_seed_usually_differs(snap):
    a = sample_tickers(10, seed=1, snapshot=snap)
    b = sample_tickers(10, seed=2, snapshot=snap)
    assert a != b


def test_draw_independent_of_global_rng(snap):
    np.random.seed(0); np.random.random(100)
    a = sample_tickers(10, seed=7, snapshot=snap)
    np.random.seed(999); np.random.random(5)
    b = sample_tickers(10, seed=7, snapshot=snap)
    assert a == b


# ---------- sampling semantics ----------

def test_without_replacement_distinct(snap):
    drawn = sample_tickers(40, seed=3, snapshot=snap)
    assert len(drawn) == len(set(drawn)) == 40


def test_n_exceeds_universe_raises(snap):
    with pytest.raises(ValueError, match="distinct"):
        sample_tickers(1000, seed=0, snapshot=snap)


def test_exclude_removes_ticker(snap):
    # Exclude BIG; it must never appear even though it has ~95% of the weight.
    for seed in range(20):
        drawn = sample_tickers(5, seed=seed, snapshot=snap, exclude=["BIG"])
        assert "BIG" not in drawn


def test_exclude_is_case_insensitive(snap):
    drawn = sample_tickers(5, seed=0, snapshot=snap, exclude=["big"])
    assert "BIG" not in drawn


def test_cap_weighting_favors_megacap(snap):
    # Across many single draws BIG should be picked far more than any small-cap.
    hits = sum("BIG" in sample_tickers(1, seed=s, snapshot=snap) for s in range(200))
    # BIG holds 1e6 / (1e6 + 50*1e3) ≈ 95.2% of probability mass.
    assert hits > 150   # ~190 expected; generous lower bound


def test_power_zero_is_uniform(snap):
    # With power=0 every name is equally likely; BIG should NOT dominate.
    hits = sum("BIG" in sample_tickers(1, seed=s, snapshot=snap, power=0.0) for s in range(200))
    # Uniform over 51 names => ~2% => ~4 hits; assert it's nowhere near cap-weighted.
    assert hits < 30


def test_negative_power_raises(snap):
    with pytest.raises(ValueError, match="power"):
        sample_tickers(5, seed=0, snapshot=snap, power=-1.0)


def test_zero_n_raises(snap):
    with pytest.raises(ValueError, match="n must be"):
        sample_tickers(0, seed=0, snapshot=snap)


# ---------- sampling_weights audit helper ----------

def test_sampling_weights_normalized_and_sorted(snap):
    w = sampling_weights(snapshot=snap)
    assert w["prob"].sum() == pytest.approx(1.0)
    assert w["ticker"].iloc[0] == "BIG"          # highest prob first
    assert w["prob"].iloc[0] == pytest.approx(1_000_000 / (1_000_000 + 50_000))


# ---------- dual-class dedup ----------

def _dual_class_snap():
    # GOOGL > GOOG in cap (real-world-shaped); FOX > FOXA reversed on purpose to
    # confirm dedup follows the HIGHER cap, not a hardcoded "primary" ticker.
    caps = {
        "GOOGL": 4_559.0, "GOOG": 4_513.0,
        "FOX": 28.0, "FOXA": 25.0,
        "AAPL": 3_000.0,  # unrelated control, never touched by dedup
    }
    return _snapshot(caps)


def test_dedup_keeps_higher_cap_ticker_per_group():
    df = _dedup_dual_class(_dual_class_snap())
    tickers = set(df["ticker"])
    assert tickers == {"GOOGL", "FOX", "AAPL"}  # GOOG, FOXA dropped (lower cap)


def test_dedup_follows_cap_not_a_hardcoded_side():
    # Swap which class has the higher cap and confirm the KEPT ticker flips too.
    df = pd.DataFrame({
        "ticker": ["GOOGL", "GOOG"], "market_cap": [100.0, 200.0],  # GOOG now bigger
        "snapshot_date": "2026-06-02",
    })
    out = _dedup_dual_class(df)
    assert list(out["ticker"]) == ["GOOG"]


def test_dedup_noop_when_only_one_class_present():
    caps = {"GOOGL": 4_559.0, "AAPL": 3_000.0}  # no GOOG row at all
    df = _dedup_dual_class(_snapshot(caps))
    assert set(df["ticker"]) == {"GOOGL", "AAPL"}


def test_sample_tickers_never_draws_both_classes():
    df = _dual_class_snap()
    for seed in range(50):
        drawn = set(sample_tickers(3, seed=seed, snapshot=df, power=0.0))
        for group in DUAL_CLASS_ISSUERS:
            assert len(drawn & group) <= 1


def test_sampling_weights_dedups_too():
    w = sampling_weights(snapshot=_dual_class_snap())
    assert set(w["ticker"]) == {"GOOGL", "FOX", "AAPL"}
