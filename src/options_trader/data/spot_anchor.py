"""
spot_anchor.py — production spot-anchor sources for the daily run.

WHICH price the forecast anchors on is a forecasting choice, separable from the
forecaster itself (which just consumes a `spot`). This names that choice so it is
explicit, injectable, and testable — the production-side mirror of the backtest's
anchor sources in ``backtest/anchor.py``.

The Session-9 anchor experiment measured these two choices on the 20-ticker
standardized set and ACCEPTED anchoring on the fresh price over the stale last
close (+0.075 mean log score, p=7.6e-06, 18/20 win). ``live_else_close`` is the
default; ``last_close`` is kept for the off-hours / no-creds fallback and for
parity testing.

    backtest anchor (sim)        production anchor (live)
    ---------------------        ------------------------
    intraday_random        <->   live_else_close   (anchor on the freshest price)
    stale_close            <->   last_close        (anchor on yesterday's close)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from options_trader.data.history import get_latest_price
from options_trader.data.stock_return_ts import StockReturnTS


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchoredSpot:
    """The resolved anchor price plus where it came from (for the run report)."""

    spot: float
    source: str   # "live" | "last_close"


# A spot anchor maps (ticker, history) -> AnchoredSpot.
SpotAnchor = Callable[[str, StockReturnTS], AnchoredSpot]


def last_close_anchor(ticker: str, ts: StockReturnTS) -> AnchoredSpot:
    """Anchor on the last completed daily close (no network)."""
    return AnchoredSpot(spot=float(ts.close[-1]), source="last_close")


def live_else_close_anchor(ticker: str, ts: StockReturnTS) -> AnchoredSpot:
    """Anchor on today's live trade; fall back to the last close when unavailable.

    Backtest-validated default (Session 9). The fallback keeps the run working
    off-hours / without creds / on network failure (``get_latest_price`` returns
    None on any such failure rather than raising).
    """
    last_close = float(ts.close[-1])
    live = get_latest_price(ticker)
    if live is None:
        logger.warning(
            "%s: no live price; anchoring forecast on last close %.2f", ticker, last_close
        )
        return AnchoredSpot(spot=last_close, source="last_close")
    return AnchoredSpot(spot=float(live), source="live")
