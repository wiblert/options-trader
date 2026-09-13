"""
anchor.py — injectable spot-anchor sources for the rolling backtest.

The forecaster stays a PURE (ReturnDistribution, spot) -> PriceDistribution map;
WHICH price we anchor on is a separate, swappable decision. In production that
choice (live latest trade vs last completed close) lives in run_daily; here we
expose the same choice as named, no-I/O functions so the rolling log-score
backtest can *measure* whether anchoring on today's intraday price beats
anchoring on the last completed daily close — gated by the standing decision
criterion like any other forecaster change.

Timeline modelled (one rolling iteration t):
    We are running INTRADAY on the day AFTER the last completed close.
        prev_close  = close[t]          last COMPLETED daily close (the stale anchor)
        today's bar = OHLC at index t+1 (the day we're trading on)
    Both anchors forecast the SAME horizon to the SAME realised target
    (close[t+horizon]); they differ ONLY in the starting price:
        - stale_close   : grid-aligned but up to a full day old.
        - intraday      : fresh, but off the close grid by a partial day.
    The log score decides which error is smaller — that is the whole experiment.

We have no intraday ticks in the daily backtest, so the intraday price is
SYNTHESISED from the daily OHLC bar (uniform in [low, high]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AnchorContext:
    """Per-iteration inputs an anchor source may use. All prices are positive."""

    prev_close: float       # close[t]   — last COMPLETED daily close
    today_open: float       # open[t+1]
    today_high: float       # high[t+1]
    today_low: float        # low[t+1]
    today_close: float      # close[t+1] — NOT observable intraday; diagnostics/tests only
    rng: np.random.Generator
    # Completed daily closes up to AND INCLUDING prev_close (newest last). All
    # entries are <= index t, so any moving average over this window is strictly
    # point-in-time (never touches today_close or anything in the future). Empty
    # by default so anchors that don't need history (stale/intraday) and existing
    # callers/tests keep working; MA anchors require it to be populated.
    trailing_closes: np.ndarray = field(default_factory=lambda: np.empty(0))


# An anchor source maps a context to the spot price to anchor the forecast on.
AnchorFn = Callable[[AnchorContext], float]


def stale_close_anchor(ctx: AnchorContext) -> float:
    """Incumbent: anchor on the last COMPLETED daily close (yesterday's print)."""
    return ctx.prev_close


def intraday_uniform_anchor(ctx: AnchorContext) -> float:
    """Candidate: a random intraday price for today, drawn uniformly in [low, high].

    Synthesises an intraday transaction price from the daily bar (no intraday
    ticks in the backtest). Uniform-in-range is the literal "a random price
    observed during the day" — deliberately agnostic about WHERE in the day the
    trade lands, so the experiment measures the value of a fresh anchor, not a
    guess about intraday timing.
    """
    return float(ctx.rng.uniform(ctx.today_low, ctx.today_high))


# Registry the anchor experiment iterates over. Keys double as report labels.
ANCHOR_FNS: dict[str, AnchorFn] = {
    "stale_close": stale_close_anchor,
    "intraday_random": intraday_uniform_anchor,
}


# ---------- moving-average blend anchors (Hypothesis 1) ----------
#
# Question: is a single fresh print a noisy estimate of "where the stock is",
# and does smoothing it toward a short trailing average improve the forecast?
# We test it as an anchor swap because the anchor is literally the center the
# forecast distribution scales off (terminal price ~= anchor * exp(returns)).
# Each blend mixes the SAME fresh intraday price intraday_random uses with a
# trailing close MA, so the paired log-score difference isolates the smoothing.


def _trailing_ma(ctx: AnchorContext, window: int) -> float:
    """Mean of the last `window` COMPLETED closes (point-in-time; <= index t).

    Uses min(window, available) so early iterations don't fail; in the rolling
    backtest the training window is long, so the full window is always present.
    """
    closes = ctx.trailing_closes
    if closes.size == 0:
        raise ValueError(
            "moving-average anchor requires AnchorContext.trailing_closes to be "
            "populated (closes up to and including prev_close)"
        )
    w = min(window, closes.size)
    return float(np.mean(closes[-w:]))


def make_ma_blend_anchor(window: int, weight_latest: float) -> AnchorFn:
    """Anchor = weight_latest * fresh_intraday + (1 - weight_latest) * MA(window).

    weight_latest in [0, 1]: 1.0 == pure fresh price (identical to
    intraday_random), 0.0 == pure trailing MA. The fresh component is drawn the
    SAME way intraday_uniform_anchor draws it (one uniform in [low, high]) and is
    drawn UNCONDITIONALLY (even when its weight is 0) so every arm consumes
    exactly one RNG value per iteration — keeping the arms on common random
    numbers so the paired difference isolates the blend, not the draw sequence.
    """
    if not 0.0 <= weight_latest <= 1.0:
        raise ValueError(f"weight_latest must be in [0, 1], got {weight_latest}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    def anchor(ctx: AnchorContext) -> float:
        latest = float(ctx.rng.uniform(ctx.today_low, ctx.today_high))
        ma = _trailing_ma(ctx, window)
        return weight_latest * latest + (1.0 - weight_latest) * ma

    return anchor


# The headline candidate the user asked for (50/50 latest + 7-day MA), plus a
# 20-day variant and a pure-MA variant to map the smoothing curve in one run.
ma7_blend_anchor = make_ma_blend_anchor(window=7, weight_latest=0.5)
ma20_blend_anchor = make_ma_blend_anchor(window=20, weight_latest=0.5)
ma7_only_anchor = make_ma_blend_anchor(window=7, weight_latest=0.0)


# Separate registry so the validated stale-vs-intraday experiment (and its
# test_registry_keys assertion) is untouched. Incumbent (intraday_random) first
# so it's the natural baseline column in the report.
MA_BLEND_ANCHOR_FNS: dict[str, AnchorFn] = {
    "intraday_random": intraday_uniform_anchor,
    "ma7_blend": ma7_blend_anchor,
    "ma20_blend": ma20_blend_anchor,
    "ma7_only": ma7_only_anchor,
}
