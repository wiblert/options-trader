"""
manage_positions.py — the daily EXIT pass over open option positions.

Companion to run_daily.py (the ENTRY pass). run_daily opens positions; this
reprices the ones we already hold and decides hold-vs-sell:

    for each open option position:
        parse the OCC symbol -> underlying, expiry, type, strike
        history -> forecast the underlying's price dist AT THE OPTION'S EXPIRY
        pull the contract's live quote -> value it (same OptionValuer)
        compare our fair value to the current BID:
            if fair_value is significantly BELOW the bid -> the market is paying
            more than the option is worth to us -> SELL-to-close (limit at bid)
            otherwise -> HOLD

The "significantly below" test uses the relative gap (bid - fair_value) / bid
against `sell_threshold` (default 15%). Rationale: we hold a long option worth
`fair_value` to us under our forecast; if a buyer will pay materially more than
that (`bid`), we take it. Sunk cost (what we paid) is correctly ignored — only
our forward value vs the current bid matters.

Safety mirrors run_daily: DRY-RUN by default (requires --live to submit), paper
broker only, per-position error isolation, held-to-expiry forecast invariant.
This is an EXIT-only complement; it never opens a position.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

import numpy as np

from options_trader.data.history import get_history, HistoryError
from options_trader.data.occ import parse_occ_symbol
from options_trader.data.options_chain import get_option_chain, OptionsChainError
from options_trader.data.spot_anchor import SpotAnchor, live_else_close_anchor
from options_trader.forecast.factories import (
    ForecastContext,
    ForecasterFactory,
    bootstrap_factory,
    event_days_in_horizon,
)
from options_trader.forecast.return_distribution import ReturnDistribution
from options_trader.valuation.option_valuer import (
    OptionContract,
    OptionType,
    OptionValuation,
    OptionValuer,
)
from options_trader.execution.broker import AlpacaBroker, OrderResult, PositionSnapshot
from options_trader.config import DEFAULT_N_PATHS, DEFAULT_RISK_FREE_RATE
from options_trader.run_daily import _ticker_seed


logger = logging.getLogger(__name__)

DEFAULT_SELL_THRESHOLD = 0.15   # sell if fair_value is >=15% below the bid


@dataclass
class PositionReview:
    """The repricing + decision for one held option position."""

    symbol: str
    qty: int
    underlying: Optional[str] = None
    option_type: Optional[OptionType] = None
    strike: Optional[float] = None
    expiry: Optional[date] = None
    spot: Optional[float] = None
    spot_source: Optional[str] = None
    horizon: Optional[int] = None
    events_in_horizon: Optional[int] = None
    bid: Optional[float] = None
    fair_value: Optional[float] = None      # our per-share P-measure EV
    gap_pct: Optional[float] = None         # (bid - fair_value) / bid; >0 ⇒ overpriced
    avg_entry_price: Optional[float] = None
    unrealized_pl: Optional[float] = None
    action: str = "HOLD"                    # "SELL" | "HOLD" | "SKIP" | "ERROR"
    reason: Optional[str] = None
    order: Optional[OrderResult] = None


def decide_action(
    fair_value: Optional[float], bid: Optional[float], sell_threshold: float
) -> tuple[str, Optional[float], str]:
    """Pure hold/sell rule. Returns (action, gap_pct, reason).

    Sell only when there is a live bid AND our fair value is at least
    `sell_threshold` below it (relative to the bid). Otherwise hold.
    """
    if bid is None or bid <= 0:
        return "HOLD", None, "no live bid"
    if fair_value is None:
        return "HOLD", None, "no fair value"
    gap = (bid - fair_value) / bid
    if gap >= sell_threshold:
        return "SELL", gap, f"fair {fair_value:.2f} is {gap:.0%} below bid {bid:.2f}"
    return "HOLD", gap, f"fair {fair_value:.2f} within {sell_threshold:.0%} of bid {bid:.2f}"


def _quote_for_symbol(occ, valuation_date: date) -> Optional[OptionContract]:
    """Pull the live quote for exactly this held contract (match by symbol)."""
    # Tight strike window around the exact strike, then match the OCC symbol.
    pad = max(0.01, occ.strike * 0.001)
    chain = get_option_chain(
        occ.underlying,
        option_type=occ.option_type,
        expiration_gte=occ.expiry, expiration_lte=occ.expiry,
        strike_gte=occ.strike - pad, strike_lte=occ.strike + pad,
    )
    for c in chain:
        if c.symbol.upper() == occ.symbol:
            return c
    return None


def review_position(
    position: PositionSnapshot,
    *,
    run_date: date,
    valuer: OptionValuer,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = 42,
    spot_anchor: SpotAnchor = live_else_close_anchor,
    forecaster_factory: ForecasterFactory = bootstrap_factory,
    forecast_sink: Optional[Callable] = None,
) -> PositionReview:
    """Reprice one held option position and decide hold vs sell.

    `forecast_sink`, if given, is called with
    ``(underlying, ts, pdist, spot, horizon)`` right after the forecast — the
    review webapp uses it to capture the exact PriceDistribution for charting.
    """
    review = PositionReview(
        symbol=position.symbol, qty=int(position.qty),
        avg_entry_price=position.avg_entry_price, unrealized_pl=position.unrealized_pl,
    )
    try:
        occ = parse_occ_symbol(position.symbol)
        review.underlying = occ.underlying
        review.option_type = occ.option_type
        review.strike = occ.strike
        review.expiry = occ.expiry

        if occ.expiry <= run_date:
            review.action, review.reason = "SKIP", "expired / expires today"
            return review

        ts = get_history(occ.underlying)
        anchored = spot_anchor(occ.underlying, ts)
        review.spot, review.spot_source = anchored.spot, anchored.source

        horizon = int(np.busday_count(run_date, occ.expiry))
        review.horizon = horizon
        if horizon < 1:
            review.action, review.reason = "SKIP", "expiry < 1 trading day out"
            return review

        rd = ReturnDistribution(np.diff(np.log(ts.close)), label=occ.underlying)
        ctx = ForecastContext(
            ticker=occ.underlying, ts=ts, rd=rd, horizon=horizon,
            run_date=run_date, n_paths=n_paths, seed=_ticker_seed(seed, occ.underlying),
        )
        forecaster = forecaster_factory(ctx)
        review.events_in_horizon = event_days_in_horizon(forecaster)
        pdist = forecaster.forecast(horizon, anchored.spot)
        if forecast_sink is not None:
            forecast_sink(occ.underlying, ts, pdist, anchored.spot, horizon)

        contract = _quote_for_symbol(occ, run_date)
        if contract is None:
            review.action, review.reason = "HOLD", "contract not found in live chain"
            return review

        v: OptionValuation = valuer.value(pdist, contract, spot=anchored.spot, valuation_date=run_date)
        review.bid = contract.bid
        review.fair_value = v.fair_value

        action, gap, reason = decide_action(v.fair_value, contract.bid, sell_threshold)
        review.action, review.gap_pct, review.reason = action, gap, reason
    except (HistoryError, OptionsChainError, ValueError) as exc:
        review.action, review.reason = "ERROR", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # never let one position abort the batch
        logger.exception("Unexpected error reviewing %s", position.symbol)
        review.action, review.reason = "ERROR", f"{type(exc).__name__}: {exc}"
    return review


@dataclass
class ManageRunResult:
    run_date: date
    dry_run: bool
    reviews: list[PositionReview]

    @property
    def sells(self) -> list[PositionReview]:
        return [r for r in self.reviews if r.action == "SELL"]


def manage_positions(
    broker: AlpacaBroker,
    *,
    run_date: Optional[date] = None,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = 42,
    spot_anchor: SpotAnchor = live_else_close_anchor,
    forecaster_factory: Optional[ForecasterFactory] = None,
) -> ManageRunResult:
    """Reprice every open OPTION position, then (unless dry-run) submit sell-to-close
    limit orders for the ones our forecast says are overpriced by `sell_threshold`."""
    run_date = run_date or date.today()
    if forecaster_factory is None:
        from options_trader.run_daily import _default_forecaster_factory
        forecaster_factory = _default_forecaster_factory()

    valuer = OptionValuer(risk_free_rate=risk_free_rate)

    option_positions = [
        p for p in broker.get_positions() if "option" in p.asset_class.lower()
    ]

    # Symbols that already have a resting SELL — don't resubmit (re-run idempotency).
    try:
        resting_sells = {
            o.symbol for o in broker.get_open_orders() if o.side.lower() == "sell"
        }
    except Exception:  # best-effort; never fatal
        logger.warning("Could not read open orders; assuming none", exc_info=True)
        resting_sells = set()

    reviews: list[PositionReview] = []
    for p in option_positions:
        logger.info("Reviewing %s", p.symbol)
        reviews.append(review_position(
            p, run_date=run_date, valuer=valuer, sell_threshold=sell_threshold,
            n_paths=n_paths, seed=seed, spot_anchor=spot_anchor,
            forecaster_factory=forecaster_factory,
        ))

    # Submit a sell-to-close at the bid for each SELL verdict, unless one is already
    # resting (a same-day re-run must not stack a second sell on the same contract).
    for r in reviews:
        if r.action == "SELL" and r.bid is not None:
            if r.symbol in resting_sells:
                r.reason = f"{r.reason or ''} | resting sell exists, not resubmitted".strip(" |")
                continue
            r.order = broker.submit_sell(r.symbol, r.qty, limit_price=r.bid)

    return ManageRunResult(run_date=run_date, dry_run=broker.dry_run, reviews=reviews)


# ----------------------------- CLI -----------------------------

def _print_summary(result: ManageRunResult) -> None:
    from tabulate import tabulate

    mode = "DRY-RUN (no orders placed)" if result.dry_run else "LIVE PAPER"
    print(f"\n=== manage_positions {result.run_date} | {mode} ===\n")

    rows = []
    for r in sorted(result.reviews, key=lambda x: x.symbol):
        ev = "" if not r.events_in_horizon else str(r.events_in_horizon)
        order = r.order.status.value if r.order else "—"
        rows.append([
            r.symbol, r.qty,
            f"{r.spot:.2f}" if r.spot is not None else "—",
            str(r.expiry) if r.expiry else "—", ev,
            f"{r.bid:.2f}" if r.bid is not None else "—",
            f"{r.fair_value:.2f}" if r.fair_value is not None else "—",
            f"{r.gap_pct:+.0%}" if r.gap_pct is not None else "—",
            f"{r.unrealized_pl:+,.0f}" if r.unrealized_pl is not None else "—",
            r.action, order,
            (r.reason or "")[:38],
        ])
    print(tabulate(
        rows,
        headers=["symbol", "qty", "spot", "expiry", "ev", "bid", "fair", "gap",
                 "unrl P&L", "action", "order", "reason"],
        tablefmt="github",
    ))

    n = len(result.reviews)
    n_sell = sum(1 for r in result.reviews if r.action == "SELL")
    n_hold = sum(1 for r in result.reviews if r.action == "HOLD")
    n_skip = sum(1 for r in result.reviews if r.action == "SKIP")
    n_err = sum(1 for r in result.reviews if r.action == "ERROR")
    print(f"\n{n} option positions | {n_sell} SELL | {n_hold} HOLD | {n_skip} SKIP | {n_err} ERROR")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Daily EXIT pass: reprice open option positions, sell the overpriced.")
    ap.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD,
                    help="sell if fair value is >= this fraction below the bid (default 0.15)")
    ap.add_argument("--n-paths", type=int, default=DEFAULT_N_PATHS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-events", action="store_true",
                    help="use the plain bootstrap forecaster (default: event-conditioned)")
    ap.add_argument("--live", action="store_true", help="actually submit sell orders (default: dry-run)")
    args = ap.parse_args()

    broker = AlpacaBroker(dry_run=not args.live)
    factory = None
    if args.no_events:
        factory = bootstrap_factory
    result = manage_positions(
        broker, sell_threshold=args.sell_threshold,
        n_paths=args.n_paths, seed=args.seed, forecaster_factory=factory,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()
