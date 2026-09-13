"""
sampler.py — reproducible ticker sampling from a frozen snapshot.

The standardized test draws its test tickers here. Sampling is:
    * WITHOUT replacement — a test set must have distinct tickers.
    * Probability ∝ market_cap ** power — `power=0.0` is uniform (the daily-run
      default); `power=1.0` is pure cap-weighting; values between flatten the
      concentration. The standardized test uses power=1.0 for stability.
    * Seeded — same (snapshot, n, seed, power, min_market_cap, exclude) →
      identical draw, every time. This is what makes the test "standardized".
    * Dual-class deduped — see DUAL_CLASS_ISSUERS below.

Determinism note: numpy's Generator.choice(replace=False, p=...) performs weighted
sampling without replacement deterministically for a fixed seed. We never call
the global RNG, so results are independent of any other randomness in the process.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from options_trader.universe.snapshot import load_snapshot


logger = logging.getLogger(__name__)


# Known dual-class issuers: each frozenset is ONE company represented by more
# than one ticker in the S&P 500 snapshot (e.g. GOOGL/GOOG are both Alphabet).
# Sampling treats every snapshot row as an independent issuer, so without this
# both classes can be drawn into the SAME watchlist — silently doubling that
# company's effective exposure past the per-name cap. `_dedup_dual_class` keeps
# only the higher-market-cap ticker per group (self-adjusting if the cap
# ordering between classes flips, rather than hardcoding which class "wins").
# Small and static by design — dual-class S&P members are rare and don't
# change often; add a group here if a new one surfaces.
DUAL_CLASS_ISSUERS: tuple[frozenset[str], ...] = (
    frozenset({"GOOGL", "GOOG"}),   # Alphabet
    frozenset({"FOXA", "FOX"}),     # Fox Corp
    frozenset({"NWSA", "NWS"}),     # News Corp
)


def _dedup_dual_class(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the lower-market-cap ticker of each dual-class group present in `df`."""
    drop: set[str] = set()
    for group in DUAL_CLASS_ISSUERS:
        present = df[df["ticker"].isin(group)]
        if len(present) > 1:
            keep = present.loc[present["market_cap"].idxmax(), "ticker"]
            dropped = set(present["ticker"]) - {keep}
            drop |= dropped
            logger.info("Dual-class dedup: keeping %s over %s", keep, sorted(dropped))
    return df[~df["ticker"].isin(drop)] if drop else df


def sample_tickers(
    n: int,
    seed: int,
    snapshot: Optional[pd.DataFrame] = None,
    snapshot_path: Optional[Path] = None,
    power: float = 1.0,
    exclude: Iterable[str] = (),
    min_market_cap: float = 0.0,
) -> list[str]:
    """Draw `n` distinct tickers from a frozen snapshot.

    Args:
        n: number of tickers to draw.
        seed: RNG seed — fixes the draw for reproducibility.
        snapshot: pre-loaded snapshot DataFrame (columns ticker, market_cap). If
            None, loads `snapshot_path` (or the latest committed snapshot).
        snapshot_path: explicit snapshot CSV to load when `snapshot` is None.
        power: cap-weight exponent. 0.0 = uniform (daily-run default), 1.0 =
            prob ∝ market cap, between = flattened cap-weighting.
        exclude: tickers to remove from the universe before sampling (e.g. a fixed
            ticker already in the test set, so it isn't drawn again).
        min_market_cap: drop tickers with market_cap below this threshold before
            sampling. Use to filter out names with thin option chains (e.g. 10e9
            removes the bottom ~18 S&P 500 members where option liquidity is poor).

    Returns:
        List of `n` distinct ticker symbols, in draw order. Dual-class issuers
        (see DUAL_CLASS_ISSUERS) contribute at most one ticker.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if power < 0:
        raise ValueError(f"power must be >= 0, got {power}")

    df = snapshot if snapshot is not None else load_snapshot(snapshot_path)
    exclude_set = {s.upper() for s in exclude}
    df = df[~df["ticker"].str.upper().isin(exclude_set)]
    df = df[(df["market_cap"] > 0) & df["market_cap"].notna()]
    if min_market_cap > 0:
        df = df[df["market_cap"] >= min_market_cap]
    df = _dedup_dual_class(df)

    universe_size = len(df)
    if n > universe_size:
        raise ValueError(
            f"cannot draw {n} distinct tickers from a universe of {universe_size} "
            f"(after exclusions)"
        )

    weights = df["market_cap"].to_numpy(dtype=float) ** power
    probs = weights / weights.sum()

    rng = np.random.default_rng(seed)
    idx = rng.choice(universe_size, size=n, replace=False, p=probs)
    drawn = df["ticker"].to_numpy()[idx].tolist()

    logger.info(
        "Sampled %d tickers (seed=%d, power=%.2f) from universe of %d: %s",
        n, seed, power, universe_size, ", ".join(drawn),
    )
    return drawn


def sampling_weights(
    snapshot: Optional[pd.DataFrame] = None,
    snapshot_path: Optional[Path] = None,
    power: float = 1.0,
    min_market_cap: float = 0.0,
) -> pd.DataFrame:
    """Return the universe with its normalized sampling probabilities.

    Useful for documenting/auditing how concentrated the cap-weighting is (e.g.
    "top 10 names hold X% of sampling probability"). Mirrors `sample_tickers`'s
    dual-class dedup so the reported probabilities match what actually gets drawn.
    """
    df = (snapshot if snapshot is not None else load_snapshot(snapshot_path)).copy()
    df = df[(df["market_cap"] > 0) & df["market_cap"].notna()].copy()
    if min_market_cap > 0:
        df = df[df["market_cap"] >= min_market_cap].copy()
    df = _dedup_dual_class(df)
    w = df["market_cap"].to_numpy(dtype=float) ** power
    df["prob"] = w / w.sum()
    return df.sort_values("prob", ascending=False).reset_index(drop=True)
