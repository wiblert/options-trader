"""
review_service.py — build the interactive daily-review payload for the web UI.

This is the human-in-the-loop counterpart to `run_daily`. `run_daily` plans every
ticker and *executes automatically*; this module runs the SAME pipeline but stops
short of placing any order, packaging the result as JSON the review page renders:

    EXIT  — reprice each open option position (manage_positions.review_position)
    ENTRY — plan the best BUY call/put per ticker (run_daily.plan_ticker), then
            apply the portfolio caps (PortfolioConstructor.allocate) for the
            recommended set + the gross/Kelly budget

Nothing here submits. Execution happens later, one item at a time, when the user
approves it — the server holds the computed `PositionReview` / `Candidate` objects
(see `ReviewSession`) and calls `broker.submit_sell` / `broker.execute` then.

Faithful charts: the displayed forecast distribution is the *actual* production
`PriceDistribution` used for the recommendation (GARCH-FHS by default), captured
via the `forecast_sink` hook on `plan_ticker` / `review_position` — never a
bootstrap re-derivation. The chart payload mirrors `forecast_service`'s KDE/grid
approach so the page draws history + strike line + the forecast distribution +
spot + expected value with no extra recompute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.execution.broker import AlpacaBroker, OrderResult
from options_trader.forecast.factories import ForecasterFactory
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.manage_positions import (
    DEFAULT_SELL_THRESHOLD,
    PositionReview,
    review_position,
)
from options_trader.config import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_FRACTION,
    DEFAULT_N_PATHS,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TARGET_DTE,
)
from options_trader.run_daily import (
    Candidate,
    _committed_types_by_underlying,
    _default_forecaster_factory,
    _open_option_positions,
    _open_orders,
    plan_ticker,
)
from options_trader.portfolio.constructor import PortfolioCaps, PortfolioConstructor
from options_trader.sizing.kelly import KellySizer
from options_trader.valuation.option_valuer import OptionValuer


GRID_POINTS = 200
CLIP_Q = (0.01, 0.99)
DEFAULT_HISTORY_BARS = 120


# --------------------------------------------------------------------------- #
#  Session object (returned to the server, NOT serialised to the client)
# --------------------------------------------------------------------------- #


@dataclass
class ReviewSession:
    """One computed review run.

    `payload` is the JSON-serialisable dict the client renders. `sells` / `buys`
    map each item_id to the live python object so the server can execute exactly
    what was reviewed (precise qty / limit / sizing) when the user approves it.
    `deployed_so_far` tracks premium committed by approved buys this session, so
    the budget meter stays honest as capital is put out (the Kelly-sizing view).
    """

    payload: dict
    sells: dict = field(default_factory=dict)   # item_id -> PositionReview
    buys: dict = field(default_factory=dict)     # item_id -> Candidate
    bankroll: float = 0.0
    gross_budget: float = 0.0
    current_premium: float = 0.0
    deployed_so_far: float = 0.0

    def budget_dict(self) -> dict:
        return _budget_dict(
            self.bankroll, self.gross_budget, self.current_premium, self.deployed_so_far
        )


def _capture():
    """A one-slot forecast sink: returns (holder, sink). The sink stores the
    forecast inputs the chart needs; one is used per plan_ticker / review_position
    call so concurrent same-underlying reviews never clobber each other."""
    holder: dict = {}

    def sink(_ticker, ts, pdist, spot, horizon):
        holder["ts"], holder["pdist"] = ts, pdist
        holder["spot"], holder["horizon"] = spot, horizon

    return holder, sink


# --------------------------------------------------------------------------- #
#  Build the session
# --------------------------------------------------------------------------- #


def build_review_session(
    broker: AlpacaBroker,
    tickers: list[str],
    *,
    run_date: Optional[date] = None,
    bankroll: Optional[float] = None,
    caps: Optional[PortfolioCaps] = None,
    forecaster_factory: Optional[ForecasterFactory] = None,
    target_dte: int = DEFAULT_TARGET_DTE,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_fraction: float = DEFAULT_MAX_FRACTION,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = 42,
    strike_window: float = 0.08,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    history_bars: int = DEFAULT_HISTORY_BARS,
    skip_manage: bool = False,
) -> ReviewSession:
    """Run the daily pipeline WITHOUT executing and return a reviewable session.

    Mirrors `run_daily`'s inputs/defaults so the recommendations and sizing match
    a `run_daily --tickers ...` dry-run on the same names.
    """
    run_date = run_date or date.today()
    if bankroll is None:
        bankroll = broker.get_account_snapshot().equity
    if forecaster_factory is None:
        forecaster_factory = _default_forecaster_factory()

    valuer = OptionValuer(risk_free_rate=risk_free_rate)
    sizer = KellySizer(kelly_fraction=kelly_fraction, max_fraction=max_fraction)

    positions = _open_option_positions(broker)
    open_orders = _open_orders(broker)
    current_premium = sum(p.market_value for p in positions)
    held_by_underlying = _committed_types_by_underlying(positions, open_orders)
    resting_sells = {o.symbol for o in open_orders if o.side.lower() == "sell"}

    # --- EXIT pass: reprice held positions (no submit). ---
    sells_json: list[dict] = []
    sells_map: dict = {}
    if not skip_manage:
        for p in positions:
            holder, sink = _capture()
            review = review_position(
                p, run_date=run_date, valuer=valuer, sell_threshold=sell_threshold,
                n_paths=n_paths, seed=seed, forecaster_factory=forecaster_factory,
                forecast_sink=sink,
            )
            item_id = f"sell:{review.symbol}"
            sells_map[item_id] = review
            sells_json.append(
                _sell_dict(review, item_id=item_id, holder=holder,
                           resting=review.symbol in resting_sells,
                           history_bars=history_bars)
            )

    # --- ENTRY pass: plan each ticker (no execute), then allocate. ---
    buy_holders: dict = {}   # ticker -> capture holder (shared across its candidates)
    plans = []
    for ticker in tickers:
        holder, sink = _capture()
        plan = plan_ticker(
            ticker, run_date=run_date, bankroll=bankroll, valuer=valuer, sizer=sizer,
            target_dte=target_dte, n_paths=n_paths, seed=seed, strike_window=strike_window,
            held_types=held_by_underlying.get(ticker.upper(), frozenset()),
            forecaster_factory=forecaster_factory, forecast_sink=sink,
        )
        plans.append(plan)
        buy_holders[ticker] = holder

    candidates = [c for p in plans for c in p.candidates]
    allocation = PortfolioConstructor(caps).allocate(candidates, bankroll, current_premium)

    buys_json: list[dict] = []
    buys_map: dict = {}
    # accepted first, then by expected log growth — the order the user should review.
    ordered = sorted(
        [(p, c) for p in plans for c in p.candidates],
        key=lambda pc: (pc[1].allocated, pc[1].sizing.expected_log_growth),
        reverse=True,
    )
    for plan, cand in ordered:
        item_id = f"buy:{cand.valuation.contract.symbol}"
        buys_map[item_id] = cand
        buys_json.append(
            _buy_dict(cand, item_id=item_id, holder=buy_holders.get(plan.ticker, {}),
                      history_bars=history_bars)
        )

    # Tickers that produced no BUY signal (informational; not reviewable items).
    no_buy = [
        {"ticker": p.ticker, "spot": p.spot, "expiry": _iso(p.expiry),
         "error": p.error} for p in plans if not p.candidates
    ]

    payload = {
        "run_date": run_date.isoformat(),
        "dry_run": broker.dry_run,
        "budget": _budget_dict(bankroll, allocation.gross_budget, current_premium, 0.0),
        "recommended_deploy": allocation.accepted_cost,
        "hedge_haircut_applied": allocation.hedge_haircut_applied,
        "sells": sells_json,
        "buys": buys_json,
        "no_buy": no_buy,
        "counts": {
            "positions": len(sells_json),
            "sell_recommended": sum(1 for s in sells_json if s["recommended"]),
            "buy_candidates": len(buys_json),
            "buy_recommended": sum(1 for b in buys_json if b["allocated"]),
        },
    }

    return ReviewSession(
        payload=payload, sells=sells_map, buys=buys_map,
        bankroll=bankroll, gross_budget=allocation.gross_budget,
        current_premium=current_premium, deployed_so_far=0.0,
    )


# --------------------------------------------------------------------------- #
#  Execution (called by the server on approve) — mutates the session budget
# --------------------------------------------------------------------------- #


def execute_item(session: ReviewSession, item_id: str, *, override: bool = False) -> dict:
    """Execute one approved review item via the broker; update the budget meter.

    Sells submit a sell-to-close at the bid; buys go through `broker.execute`
    (which applies its own account / buying-power guards). A buy that would push
    deployed premium past the gross budget is refused unless `override` is set —
    that's the Kelly-budget guardrail surfaced to the click.
    """
    if item_id in session.sells:
        return _execute_sell(session, item_id)
    if item_id in session.buys:
        return _execute_buy(session, item_id, override=override)
    return {"error": f"unknown item_id {item_id!r}"}


def _execute_sell(session: ReviewSession, item_id: str) -> dict:
    review: PositionReview = session.sells[item_id]
    if review.bid is None or review.bid <= 0:
        return {"error": "no live bid to sell into", "budget": session.budget_dict()}
    broker = _broker_of(session)
    order = broker.submit_sell(review.symbol, review.qty, limit_price=review.bid)
    return {"order": _order_dict(order), "budget": session.budget_dict()}


def _execute_buy(session: ReviewSession, item_id: str, *, override: bool) -> dict:
    cand: Candidate = session.buys[item_id]
    cost = cand.sizing.cost
    remaining = session.gross_budget - session.current_premium - session.deployed_so_far
    if not override and cost > remaining + 1e-6:
        return {
            "blocked": "over_budget",
            "message": (
                f"buying this would deploy ${cost:,.0f} but only ${max(0.0, remaining):,.0f} "
                f"of the Kelly gross budget remains"
            ),
            "budget": session.budget_dict(),
        }
    broker = _broker_of(session)
    order = broker.execute(cand.sizing)
    cand.order = order
    # Only count premium actually committed (submitted or dry-run rehearsal).
    if order.status.value in ("SUBMITTED", "DRY_RUN"):
        session.deployed_so_far += cost
    return {"order": _order_dict(order), "budget": session.budget_dict()}


def _broker_of(session: ReviewSession) -> AlpacaBroker:
    # The broker is attached to the session by the server (see server.py).
    broker = getattr(session, "_broker", None)
    if broker is None:
        raise RuntimeError("session has no broker attached")
    return broker


# --------------------------------------------------------------------------- #
#  Serialisers
# --------------------------------------------------------------------------- #


def _iso(d) -> Optional[str]:
    return d.isoformat() if d is not None else None


def _budget_dict(bankroll: float, gross_budget: float, current_premium: float,
                 deployed_so_far: float) -> dict:
    remaining = max(0.0, gross_budget - current_premium - deployed_so_far)
    return {
        "bankroll": bankroll,
        "gross_budget": gross_budget,
        "gross_fraction": (gross_budget / bankroll) if bankroll else 0.0,
        "current_premium": current_premium,
        "deployed_so_far": deployed_so_far,
        "remaining": remaining,
    }


def _order_dict(order: OrderResult) -> dict:
    return {
        "status": order.status.value,
        "symbol": order.symbol,
        "qty": order.qty,
        "side": order.side,
        "limit_price": order.limit_price,
        "order_id": order.order_id,
        "broker_status": order.broker_status,
        "reason": order.reason,
    }


def _sell_dict(review: PositionReview, *, item_id: str, holder: dict,
               resting: bool, history_bars: int) -> dict:
    chart = None
    if "pdist" in holder and review.strike is not None:
        chart = _chart_payload(
            holder["ts"], holder["pdist"], spot=holder["spot"], strike=review.strike,
            breakeven=None, fair_value=review.fair_value, prob_itm=None,
            horizon_date=_iso(review.expiry), history_bars=history_bars,
        )
    return {
        "item_id": item_id,
        "kind": "sell",
        "symbol": review.symbol,
        "qty": review.qty,
        "underlying": review.underlying,
        "option_type": review.option_type.value if review.option_type else None,
        "strike": review.strike,
        "expiry": _iso(review.expiry),
        "spot": review.spot,
        "spot_source": review.spot_source,
        "horizon": review.horizon,
        "events_in_horizon": review.events_in_horizon,
        "bid": review.bid,
        "fair_value": review.fair_value,
        "gap_pct": review.gap_pct,
        "avg_entry_price": review.avg_entry_price,
        "unrealized_pl": review.unrealized_pl,
        "action": review.action,
        "reason": review.reason,
        "recommended": review.action == "SELL",
        "resting": resting,
        "chart": chart,
    }


def _buy_dict(cand: Candidate, *, item_id: str, holder: dict, history_bars: int) -> dict:
    v = cand.valuation
    c = v.contract
    s = cand.sizing
    d = cand.decomposition
    chart = None
    if "pdist" in holder:
        chart = _chart_payload(
            holder["ts"], holder["pdist"], spot=v.spot, strike=c.strike,
            breakeven=v.breakeven, fair_value=v.fair_value, prob_itm=v.prob_itm,
            horizon_date=_iso(c.expiry), history_bars=history_bars,
        )
    return {
        "item_id": item_id,
        "kind": "buy",
        "symbol": c.symbol,
        "underlying": c.underlying,
        "option_type": c.option_type.value,
        "strike": c.strike,
        "expiry": _iso(c.expiry),
        "spot": v.spot,
        "days_to_expiry": v.days_to_expiry,
        "ask": v.buy_price,
        "bid": v.sell_price,
        "fair_value": v.fair_value,
        "expected_payoff": v.expected_payoff,
        "prob_itm": v.prob_itm,
        "edge_buy": v.edge_buy,
        "edge_pct_buy": v.edge_pct_buy,
        "breakeven": v.breakeven,
        "recommendation": v.recommendation.value,
        "forecast_mean": d.forecast_mean,
        "forward": d.forward,
        "drift_edge": d.drift_edge,
        "shape_edge": d.shape_edge,
        "drift_share": d.drift_share,
        "sizing": {
            "n_contracts": s.n_contracts,
            "cost": s.cost,
            "premium_per_contract": s.premium_per_contract,
            "applied_fraction": s.applied_fraction,
            "full_kelly_fraction": s.full_kelly_fraction,
            "deployed_fraction": s.deployed_fraction,
            "expected_return": s.expected_return,
            "expected_log_growth": s.expected_log_growth,
            "capped": s.capped,
            "status": s.status.value,
        },
        "allocated": cand.allocated,
        "alloc_reason": cand.alloc_reason,
        "chart": chart,
    }


def _chart_payload(
    ts: StockReturnTS,
    pdist: PriceDistribution,
    *,
    spot: float,
    strike: float,
    breakeven: Optional[float],
    fair_value: Optional[float],
    prob_itm: Optional[float],
    horizon_date: Optional[str],
    history_bars: int,
) -> dict:
    """Turn the production PriceDistribution + history into the decision-view chart.

    Produces: a trailing close-price history series, the terminal-price forecast
    distribution as a smooth PDF on a price grid (faithful to the actual forecast),
    and the scalar overlays the UI draws as lines/markers — spot (existing price),
    forecast_mean (E[S_T], the expected value of the ticker), the forecast 5/50/95%
    band, the strike, breakeven, and the option's per-share fair value.
    """
    n = len(ts.close)
    start = max(0, n - history_bars)
    hist_dates = [pd.Timestamp(d).date().isoformat() for d in ts.dates[start:]]
    hist_close = np.asarray(ts.close[start:], dtype=float).tolist()

    prices = np.asarray(pdist.prices, dtype=float)
    weights = np.asarray(pdist.weights, dtype=float)
    lo = pdist.quantile(CLIP_Q[0])
    hi = pdist.quantile(CLIP_Q[1])
    # widen the grid slightly so the strike/breakeven lines stay on-canvas
    pad = 0.04 * (hi - lo) if hi > lo else max(1.0, 0.04 * spot)
    refs = [r for r in (strike, breakeven, spot) if r is not None]
    grid_lo = min([lo - pad] + refs)
    grid_hi = max([hi + pad] + refs)
    grid = np.linspace(grid_lo, grid_hi, GRID_POINTS)
    try:
        kde = stats.gaussian_kde(prices, weights=weights)
        pdf = kde(grid).tolist()
    except Exception:  # degenerate (zero-variance) sample → fall back to a histogram
        counts, edges = np.histogram(prices, bins=40, range=(grid_lo, grid_hi),
                                     weights=weights, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        pdf = np.interp(grid, centers, counts).tolist()

    return {
        "history": {"dates": hist_dates, "close": hist_close},
        "horizon_date": horizon_date,
        "terminal": {
            "grid": grid.tolist(),
            "pdf": pdf,
            "mean": float(pdist.mean()),
            "median": pdist.quantile(0.5),
            "q05": pdist.quantile(0.05),
            "q50": pdist.quantile(0.5),
            "q95": pdist.quantile(0.95),
        },
        "spot": spot,
        "forecast_mean": float(pdist.mean()),
        "strike": strike,
        "breakeven": breakeven,
        "fair_value": fair_value,
        "prob_itm": prob_itm,
    }
