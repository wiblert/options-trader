"""
run_daily.py — the daily orchestrator.

Wires the whole pipeline together over a watchlist:

    for each ticker:
        history → ReturnDistribution → forecast price dist at the chosen expiry
        → value the chain → keep BUY signals → size (fractional Kelly)
        → (optionally) submit via the paper broker

Design choices for a safe multi-ticker run:
  * DRY-RUN by default. Submitting across many names is a big outward action; the
    CLI requires an explicit --live to actually place orders.
  * At most one CALL and one PUT per ticker — the ISSUE-1 correlated-bet guard.
    We keep only the single best call and single best put (by expected log growth)
    per name, so we never stack many correlated contracts on one underlying (e.g. 9
    puts at different strikes = one directional bet) while still allowing a genuine
    two-sided / vol view. Existing holdings count: a contract type we already hold on
    a name is excluded, so {held + ordered} ≤ one call and one put per ticker.
    This is NOT a portfolio-level exposure cap across tickers yet (still on the
    backlog); each ticker is sized against the full bankroll independently, so the
    aggregate deployed fraction is reported so the over-allocation is visible.
  * Per-ticker failures are isolated: one bad chain/history does not abort the run.

Held-to-expiry invariant is preserved: the forecast horizon is the trading-day
count to the selected expiry.
"""

from __future__ import annotations

import argparse
import logging
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from options_trader.manage_positions import ManageRunResult

from options_trader.config import (
    DEFAULT_MAX_FRACTION,
    DEFAULT_N_PATHS,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_TARGET_DTE,
)

import numpy as np

from options_trader.data.history import get_history, HistoryError
from options_trader.data.options_chain import get_option_chain, OptionsChainError
from options_trader.data.spot_anchor import SpotAnchor, live_else_close_anchor
from options_trader.data.events.composite import CompositeEventSource
from options_trader.data.events.fomc_source import FomcCalendarSource
from options_trader.data.events.manual_calendar import ManualCalendarSource
from options_trader.data.events.yfinance_source import YFinanceEarningsSource
from options_trader.forecast.factories import (
    EventBootstrapFactory,
    GarchFhsFactory,
    LiveEodPdfCache,
    OptionImpliedBetaFactory,
    BlendFactory,
    ForecastContext,
    ForecasterFactory,
    bootstrap_factory,
    garch_fhs_factory,
    event_days_in_horizon,
)
from options_trader.forecast.return_distribution import ReturnDistribution
from options_trader.valuation.option_valuer import (
    EdgeDecomposition,
    OptionType,
    OptionValuation,
    OptionValuer,
    Recommendation,
)
from options_trader.sizing.kelly import KellySizer, KellySizing, SizingStatus
from options_trader.data.occ import parse_occ_symbol
from options_trader.execution.broker import (
    AlpacaBroker,
    OrderResult,
    OrderSnapshot,
    PositionSnapshot,
)
from options_trader.portfolio.constructor import (
    AllocationResult,
    PortfolioCaps,
    PortfolioConstructor,
)


logger = logging.getLogger(__name__)

# Tickers are planned independently (their own history/chain/forecast calls), so a
# thread pool turns a multi-minute sequential 40-name run into seconds — each
# ticker is I/O-bound (network) for most of its wall time. Bounded so we don't
# hammer the data/broker APIs with a large watchlist.
DEFAULT_MAX_WORKERS = 8
DEFAULT_N_TICKERS = 40        # watchlist size when sampling
# Uniform sampling (power=0.0) — we don't favour large caps, just require options
# liquidity. The min-market-cap floor filters the thin-chain tail of the S&P 500.
DEFAULT_WATCHLIST_POWER = 0.0
# Drop S&P 500 members below this market cap before sampling. Names below ~$10B
# tend to have wide option bid-ask spreads and sparse chains. The floor removes
# ~18 of 503 universe members (bottom 4%).
DEFAULT_MIN_MARKET_CAP = 10_000_000_000
# config/manual_events.csv at the repo root (src/options_trader/run_daily.py -> parents[2]).
MANUAL_EVENTS_CSV = Path(__file__).resolve().parents[2] / "config" / "manual_events.csv"


def _default_forecaster_factory(event_window: int = 0) -> ForecasterFactory:
    """Production default: 50/50 BLEND of event-bootstrap ⊕ event-OIB (switched S18).

    A path-level probability mixture (`BlendFactory`): half the terminal-price mass
    comes from the event-conditioned bootstrap (historical/empirical), half from the
    event-conditioned Option-Implied Beta forecaster (forward option-implied via the
    SPY/IWM risk-neutral PDF mapped on by Beta + earnings-conditioned idiosyncratic
    residuals). The two draw on DIFFERENT information and win on different (vol/beta)
    names, so the mixture is the more robust expression of the signal.

    WHY the blend: S18 backtest (40 names, h=21, 1yr) — blend beats event-bootstrap
    +0.0107 (p≈6e-14, 27/40 names). The gain is largely inherited from OIB and the
    blend sits between the two arms on 33/40 (a robustness/lower-variance play) rather
    than a large outright gain; the edge is also single-window (OIB's advantage has
    been regime-dependent). Switched per user direction. See docs/results.md S18.

    Each arm degrades to plain bootstrap per-ticker if its feed fails (so a flaky
    earnings or index-options feed never aborts a ticker), and the blend of two
    degraded arms is still a valid forecast. Use --no-events to fall back to the plain
    iid bootstrap globally."""
    source = CompositeEventSource(
        YFinanceEarningsSource(),
        FomcCalendarSource(),
        ManualCalendarSource(MANUAL_EVENTS_CSV),
    )
    # Shared across every ticker in the watchlist: the OIB arm pulls the SPY/IWM
    # PDF per ticker, but run_date and (usually) horizon are the same across the
    # whole run, so caching per (symbol, run_date, horizon) turns N chain fetches
    # into ~2 (SPY + IWM) for the entire batch.
    index_pdf_cache = LiveEodPdfCache()
    return BlendFactory(
        [
            EventBootstrapFactory(source, event_window=event_window),
            OptionImpliedBetaFactory(
                index_pdf_fn=index_pdf_cache, event_source=source, event_window=event_window,
            ),
        ],
        weights=(0.5, 0.5),
    )


@dataclass
class Candidate:
    valuation: OptionValuation
    sizing: KellySizing
    decomposition: EdgeDecomposition
    allocated: bool = False                 # set by the portfolio constructor
    alloc_reason: Optional[str] = None      # 'accepted' | 'gross_budget' | 'per_name_cap'
    order: Optional[OrderResult] = None     # set only for allocated candidates


@dataclass
class TickerPlan:
    ticker: str
    spot: Optional[float] = None
    spot_source: Optional[str] = None       # "live" | "last_close" (anchor used)
    expiry: Optional[date] = None
    horizon: Optional[int] = None
    events_in_horizon: Optional[int] = None  # # event days the forecaster routed (0 if none / plain)
    forecast_mean: Optional[float] = None
    candidates: list[Candidate] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DailyRunResult:
    run_date: date
    bankroll: float
    dry_run: bool
    plans: list[TickerPlan]
    allocation: Optional[AllocationResult] = None
    manage_result: Optional["ManageRunResult"] = None

    @property
    def all_candidates(self) -> list[Candidate]:
        return [c for p in self.plans for c in p.candidates]

    @property
    def accepted_candidates(self) -> list[Candidate]:
        return [c for c in self.all_candidates if c.allocated]

    @property
    def sized_cost(self) -> float:
        """Premium the sizer wanted, before the portfolio caps."""
        return sum(c.sizing.cost for c in self.all_candidates)

    @property
    def total_cost(self) -> float:
        """Premium actually deployed after the portfolio caps."""
        return sum(c.sizing.cost for c in self.accepted_candidates)

    @property
    def aggregate_deployed_fraction(self) -> float:
        return self.total_cost / self.bankroll if self.bankroll else 0.0


def _ticker_seed(base_seed: int, ticker: str) -> int:
    """Per-ticker MC seed derived from `base_seed` and the symbol.

    A single shared seed across a whole watchlist run correlates the Monte-Carlo
    noise between names (same draw shape, different scale) — this decorrelates
    it while staying fully deterministic/reproducible given base_seed + ticker.
    """
    return base_seed + zlib.crc32(ticker.encode()) % 10_000


def _expiry_window(target_dte: int, run_date: date) -> tuple[date, date]:
    """DTE window used to discover listed expiries around target_dte.

    Upper bound reaches target_dte + 35 days so the window always spans a full
    monthly-expiry cycle (consecutive 3rd-Friday monthlies are ≤35 days apart).
    Without this, a target that lands BETWEEN two monthly expirations finds
    nothing for names that list ONLY monthlies (no weeklies) — e.g. most mid-caps —
    and they error out. The lower bound stays conservative (target − 10) so a liquid
    name with weeklies isn't pulled to an ultra-short expiry; widening the window only
    adds fallbacks, since selection is closest-to-target regardless of window width.
    """
    lo = run_date.fromordinal(run_date.toordinal() + max(1, target_dte - 10))
    hi = run_date.fromordinal(run_date.toordinal() + target_dte + 35)
    return lo, hi


def _select_expiry(
    ticker: str, chain: list, target_dte: int, run_date: date
) -> tuple[date, int]:
    """Pick the listed expiry whose DTE is closest to target_dte.

    `chain` must already span the DTE window from `_expiry_window` (the caller
    fetches it once and reuses it for valuation too — see `plan_ticker`).
    Raises OptionsChainError if no expiries are listed.
    """
    expiries = sorted({c.expiry for c in chain})
    if not expiries:
        raise OptionsChainError(f"no expiries near {target_dte} DTE for {ticker}")
    target_day = run_date.toordinal() + target_dte
    expiry = min(expiries, key=lambda e: abs(e.toordinal() - target_day))
    horizon = int(np.busday_count(run_date, expiry))
    return expiry, horizon


def plan_ticker(
    ticker: str,
    *,
    run_date: date,
    bankroll: float,
    valuer: OptionValuer,
    sizer: KellySizer,
    target_dte: int = DEFAULT_TARGET_DTE,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = 42,
    strike_window: float = 0.08,
    held_types: frozenset[OptionType] = frozenset(),
    spot_anchor: SpotAnchor = live_else_close_anchor,
    forecaster_factory: ForecasterFactory = bootstrap_factory,
    forecast_sink: Optional[Callable] = None,
) -> TickerPlan:
    """Build the (unexecuted) plan for one ticker: the best BUY call and best BUY put.

    `held_types` are option types already held on this ticker; candidates of those
    types are excluded so {held + ordered} stays ≤ one call and one put per name.

    `forecast_sink`, if given, is called once with
    ``(ticker, ts, pdist, spot, horizon)`` right after the forecast is produced —
    a no-op hook the review webapp uses to capture the exact production
    PriceDistribution for charting without re-deriving it.
    """
    plan = TickerPlan(ticker=ticker)
    try:
        ts = get_history(ticker)
        # WHICH price to anchor the forecast on is a named, swappable choice
        # (see data/spot_anchor.py). Default anchors on today's live price and
        # falls back to the last close off-hours — the backtest-validated winner.
        anchored = spot_anchor(ticker, ts)
        spot = anchored.spot
        plan.spot, plan.spot_source = spot, anchored.source

        # One chain fetch spans both the expiry-discovery window and the
        # valuation strike band, so _select_expiry and the candidate scan
        # below share it instead of double-fetching the same underlying.
        lo, hi = _expiry_window(target_dte, run_date)
        chain = get_option_chain(
            ticker,
            expiration_gte=lo, expiration_lte=hi,
            strike_gte=spot * min(0.98, 1 - strike_window),
            strike_lte=spot * max(1.02, 1 + strike_window),
        )
        expiry, horizon = _select_expiry(ticker, chain, target_dte, run_date)
        plan.expiry, plan.horizon = expiry, horizon
        if horizon < 1:
            plan.error = f"selected expiry {expiry} is <1 trading day out"
            return plan

        rd = ReturnDistribution(np.diff(np.log(ts.close)), label=ticker)
        # The forecaster is pluggable (mirrors the backtest's factory): the default
        # event-conditioned one injects upcoming earnings into the horizon; the plain
        # bootstrap is the fallback. forecast() signature is identical either way.
        ctx = ForecastContext(
            ticker=ticker, ts=ts, rd=rd, horizon=horizon,
            run_date=run_date, n_paths=n_paths, seed=_ticker_seed(seed, ticker),
        )
        forecaster = forecaster_factory(ctx)
        plan.events_in_horizon = event_days_in_horizon(forecaster)
        pdist = forecaster.forecast(horizon, spot)
        plan.forecast_mean = float(pdist.mean())
        if forecast_sink is not None:
            forecast_sink(ticker, ts, pdist, spot, horizon)

        lo_strike, hi_strike = spot * (1 - strike_window), spot * (1 + strike_window)
        for c in chain:
            if c.expiry != expiry or c.ask is None:
                continue
            if not (lo_strike <= c.strike <= hi_strike):
                continue
            v = valuer.value(pdist, c, spot=spot, valuation_date=run_date)
            if v.recommendation != Recommendation.BUY:
                continue
            s = sizer.size(pdist, v, bankroll=bankroll)
            if s.status != SizingStatus.SIZED:
                continue
            d = valuer.decompose_edge(pdist, c, spot=spot, valuation_date=run_date)
            plan.candidates.append(Candidate(valuation=v, sizing=s, decomposition=d))

        # Concentration guard (ISSUE-1): keep at most one CALL and one PUT — the
        # best of each type by expected log growth — and never a type we already
        # hold on this name. Stops stacking correlated contracts while still allowing
        # a two-sided/vol view.
        plan.candidates.sort(key=lambda x: x.sizing.expected_log_growth, reverse=True)
        best_by_type: dict[OptionType, Candidate] = {}
        for c in plan.candidates:
            t = c.valuation.contract.option_type
            if t in held_types or t in best_by_type:
                continue
            best_by_type[t] = c
        plan.candidates = list(best_by_type.values())
    except (HistoryError, OptionsChainError) as exc:
        plan.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # never let one ticker abort the batch
        logger.exception("Unexpected error planning %s", ticker)
        plan.error = f"{type(exc).__name__}: {exc}"
    return plan


def _open_option_positions(broker: AlpacaBroker) -> list[PositionSnapshot]:
    """Open OPTION positions (best-effort; never fatal to the run)."""
    try:
        return [p for p in broker.get_positions() if "option" in p.asset_class.lower()]
    except Exception:  # positions are a best-effort input, never fatal
        logger.warning("Could not read current positions; assuming none", exc_info=True)
        return []


def _open_orders(broker: AlpacaBroker) -> list[OrderSnapshot]:
    """Resting (unfilled) orders (best-effort; never fatal to the run)."""
    try:
        return broker.get_open_orders()
    except Exception:  # open orders are a best-effort input, never fatal
        logger.warning("Could not read open orders; assuming none", exc_info=True)
        return []


def _committed_types_by_underlying(
    positions: list[PositionSnapshot],
    open_orders: list[OrderSnapshot],
) -> dict[str, frozenset[OptionType]]:
    """Map each underlying to the option types we are already COMMITTED to long:
    types we hold (positions) plus types with a resting BUY order (pending opens).

    Resting SELLs are pending CLOSES, not new longs, so they don't add a type here
    (the position they close is already counted). This is what makes a same-day
    re-run idempotent for the one-call/one-put-per-name guard.
    """
    out: dict[str, set[OptionType]] = {}

    def add(symbol: str) -> None:
        try:
            occ = parse_occ_symbol(symbol)
        except ValueError:  # non-OCC / equity symbol — ignore
            return
        out.setdefault(occ.underlying.upper(), set()).add(occ.option_type)

    for p in positions:
        add(p.symbol)
    for o in open_orders:
        if o.side.lower() == "buy":
            add(o.symbol)
    return {k: frozenset(v) for k, v in out.items()}


def run_daily(
    tickers: list[str],
    broker: AlpacaBroker,
    *,
    run_date: Optional[date] = None,
    bankroll: Optional[float] = None,
    target_dte: int = DEFAULT_TARGET_DTE,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_fraction: float = DEFAULT_MAX_FRACTION,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = 42,
    strike_window: float = 0.08,
    caps: Optional[PortfolioCaps] = None,
    spot_anchor: SpotAnchor = live_else_close_anchor,
    forecaster_factory: Optional[ForecasterFactory] = None,
    sell_threshold: float = 0.15,
    skip_manage: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> DailyRunResult:
    """Plan every ticker, apply the portfolio caps, then (unless dry-run) submit.

    Flow: plan all tickers (no execution) → flatten candidates → PortfolioConstructor
    selects which to fund under the gross/direction/per-name caps → execute the
    accepted ones. Rejected candidates keep order=None and carry their alloc_reason.
    """
    run_date = run_date or date.today()
    if bankroll is None:
        bankroll = broker.get_account_snapshot().equity
    # Production default = event-conditioned forecaster (backtest-accepted). Built
    # here (not as a param default) so it isn't constructed at import time.
    if forecaster_factory is None:
        forecaster_factory = _default_forecaster_factory()

    # EXIT PASS — reprice held positions and submit sell-to-close for overpriced ones.
    # Runs before the entry pass so we don't size new positions against options we just
    # decided to exit. Lazy import avoids the circular dependency (manage_positions.py
    # imports _default_forecaster_factory from this module at function-call time).
    manage_result = None
    if not skip_manage:
        from options_trader.manage_positions import manage_positions as _manage_positions
        manage_result = _manage_positions(
            broker,
            run_date=run_date,
            sell_threshold=sell_threshold,
            risk_free_rate=risk_free_rate,
            n_paths=n_paths,
            seed=seed,
            spot_anchor=spot_anchor,
            forecaster_factory=forecaster_factory,
        )

    valuer = OptionValuer(risk_free_rate=risk_free_rate)
    sizer = KellySizer(kelly_fraction=kelly_fraction, max_fraction=max_fraction)

    # Existing holdings serve double duty: their premium is charged against the gross
    # budget, and (together with resting BUY orders) they drive the one-call/one-put-
    # per-name guard — we don't order a type we already hold OR have a pending buy for,
    # so a same-day re-run doesn't double up. Read once, up front.
    positions = _open_option_positions(broker)
    open_orders = _open_orders(broker)
    current_premium = sum(p.market_value for p in positions)
    held_by_underlying = _committed_types_by_underlying(positions, open_orders)

    # 1) plan every ticker (no orders yet). Tickers are independent (each makes its
    # own history/chain/forecast calls), so a thread pool overlaps their network
    # I/O — plan_ticker already isolates per-ticker errors, and valuer/sizer/
    # forecaster_factory are stateless/read-only across calls (LiveEodPdfCache,
    # the one piece of shared mutable state, is itself lock-protected).
    def _plan_one(ticker: str) -> TickerPlan:
        logger.info("Planning %s", ticker)
        return plan_ticker(
            ticker, run_date=run_date, bankroll=bankroll, valuer=valuer, sizer=sizer,
            target_dte=target_dte, n_paths=n_paths, seed=seed,
            strike_window=strike_window,
            held_types=held_by_underlying.get(ticker.upper(), frozenset()),
            spot_anchor=spot_anchor, forecaster_factory=forecaster_factory,
        )

    if not tickers:
        plans: list[TickerPlan] = []
    elif max_workers <= 1:
        plans = [_plan_one(t) for t in tickers]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers))) as pool:
            plans = list(pool.map(_plan_one, tickers))

    # 2) portfolio allocation across all candidates (gross + direction + per-name caps)
    candidates = [c for p in plans for c in p.candidates]
    allocation = PortfolioConstructor(caps).allocate(candidates, bankroll, current_premium)

    # 3) execute only the accepted candidates
    for cand in allocation.accepted:
        cand.order = broker.execute(cand.sizing)

    return DailyRunResult(
        run_date=run_date, bankroll=bankroll, dry_run=broker.dry_run,
        plans=plans, allocation=allocation, manage_result=manage_result,
    )


# ----------------------------- CLI -----------------------------

def _daily_watchlist_seed(run_date: date) -> int:
    """A seed that rotates once per day. Using the date ordinal makes the draw
    fresh each calendar day but stable within a day (re-runs see the same names)."""
    return run_date.toordinal()


def _default_watchlist(
    n: int,
    seed: int,
    power: float = DEFAULT_WATCHLIST_POWER,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
) -> list[str]:
    """Sample the watchlist from the frozen universe."""
    from options_trader.universe.sampler import sample_tickers
    return sample_tickers(n=n, seed=seed, power=power, min_market_cap=min_market_cap)


def _print_summary(result: DailyRunResult) -> None:
    from tabulate import tabulate

    if result.manage_result is not None:
        from options_trader.manage_positions import _print_summary as _print_exit_summary
        print("── EXIT PASS ──────────────────────────────────────────")
        _print_exit_summary(result.manage_result)
        print()
        print("── ENTRY PASS ─────────────────────────────────────────")

    mode = "DRY-RUN (no orders placed)" if result.dry_run else "LIVE PAPER"
    print(f"\n=== run_daily {result.run_date} | {mode} | bankroll ${result.bankroll:,.0f} ===\n")

    rows = []
    for p in sorted(result.plans, key=lambda x: x.ticker):
        if p.error:
            rows.append([p.ticker, "—", "—", "", "ERROR", "", "", "", "", "", p.error[:34]])
            continue
        ev = "" if not p.events_in_horizon else str(p.events_in_horizon)
        if not p.candidates:
            rows.append([p.ticker, f"{p.spot:.2f}", str(p.expiry), ev, "no BUY", "", "", "", "", ""])
            continue
        for c in p.candidates:
            ct = c.valuation.contract
            d = c.decomposition
            order = c.order.status.value if c.order else "—"
            verdict = "✓ fund" if c.allocated else f"✗ {c.alloc_reason}"
            rows.append([
                p.ticker, f"{p.spot:.2f}", str(p.expiry), ev,
                f"{ct.option_type.value} {ct.strike:.0f}",
                f"{ct.ask:.2f}", f"{c.valuation.edge_buy:+.2f}",
                f"{d.drift_share:.0%}" if d.drift_share is not None else "n/a",
                f"{c.sizing.n_contracts}", f"{c.sizing.cost:,.0f}",
                verdict, order,
            ])
    print(tabulate(
        rows,
        headers=["ticker", "spot", "expiry", "ev", "contract", "ask", "edge", "drift%", "N",
                 "cost", "alloc", "order"],
        tablefmt="github",
    ))

    n_cand = len(result.all_candidates)
    n_acc = len(result.accepted_candidates)
    n_err = sum(1 for p in result.plans if p.error)
    n_nobuy = sum(1 for p in result.plans if not p.error and not p.candidates)
    print(
        f"\n{len(result.plans)} tickers | {n_cand} sized signals | {n_acc} funded after caps | "
        f"{n_nobuy} no-BUY | {n_err} errors"
    )
    a = result.allocation
    if a is not None:
        dir_str = ", ".join(f"{k} ${v:,.0f}" for k, v in sorted(a.by_direction.items())) or "none"
        caps = PortfolioConstructor(None).caps
        gross_pct = a.gross_budget / result.bankroll if result.bankroll else 0.0
        print(
            f"sized premium ${result.sized_cost:,.0f} → deployed ${result.total_cost:,.0f} "
            f"({result.aggregate_deployed_fraction:.1%} of bankroll); "
            f"gross budget ${a.gross_budget:,.0f} ({gross_pct:.0%} of bankroll)"
        )
        hedge_note = (
            f"UNHEDGED → overall bet cut {caps.unhedged_haircut:.0%} "
            f"(one side >{caps.hedge_threshold:.0%} of premium)"
            if a.hedge_haircut_applied
            else f"hedged → no haircut (no side >{caps.hedge_threshold:.0%})"
        )
        print(f"directional split: {dir_str}  [{hedge_note}]")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Daily options pipeline over a watchlist.")
    ap.add_argument("--tickers", type=str, default=None, help="comma-separated; default = universe sample")
    ap.add_argument("--n-tickers", type=int, default=DEFAULT_N_TICKERS, help="watchlist size when sampling")
    ap.add_argument("--target-dte", type=int, default=DEFAULT_TARGET_DTE)
    ap.add_argument("--bankroll", type=float, default=None, help="default = paper account equity")
    ap.add_argument("--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION)
    ap.add_argument("--max-fraction", type=float, default=DEFAULT_MAX_FRACTION)
    ap.add_argument("--n-paths", type=int, default=DEFAULT_N_PATHS)
    ap.add_argument("--seed", type=int, default=42, help="forecaster RNG seed (Monte-Carlo paths)")
    ap.add_argument("--watchlist-seed", type=int, default=None,
                    help="ticker-sampling seed; default rotates daily (run-date ordinal)")
    ap.add_argument("--power", type=float, default=DEFAULT_WATCHLIST_POWER,
                    help="cap-weight exponent for sampling (0=uniform default, 1=cap-weighted)")
    ap.add_argument("--min-market-cap", type=float, default=DEFAULT_MIN_MARKET_CAP,
                    help="drop tickers below this market cap before sampling (default 10B)")
    ap.add_argument("--sell-threshold", type=float, default=0.15,
                    help="exit: sell if fair_value is >= this fraction below the bid (default 0.15)")
    ap.add_argument("--no-manage", action="store_true",
                    help="skip the exit pass (manage_positions); entry only")
    ap.add_argument("--no-events", action="store_true",
                    help="use the plain bootstrap forecaster (default: event-conditioned)")
    ap.add_argument("--garch-fhs", action="store_true",
                    help="use plain GARCH(1,1)-filtered FHS (backtest-accepted S11); "
                         "ignores the event calendar entirely")
    ap.add_argument("--garch-fhs-events", action="store_true",
                    help="use GARCH-FHS with event-day routing: normal days draw from the "
                         "GARCH-filtered vol-state-conditioned residual pool; event days "
                         "(earnings/FOMC) draw directly from the empirical event return "
                         "distribution, then advance the GARCH variance state. "
                         "Pending backtest acceptance.")
    ap.add_argument("--event-window", type=int, default=0,
                    help="include +/-N returns around each event when conditioning (default: 0)")
    ap.add_argument("--live", action="store_true", help="actually submit orders (default: dry-run)")
    ap.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                    help="max concurrent tickers when planning (1 = sequential)")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        wl_seed = args.watchlist_seed if args.watchlist_seed is not None \
            else _daily_watchlist_seed(date.today())
        tickers = _default_watchlist(args.n_tickers, wl_seed, args.power, args.min_market_cap)
        print(f"watchlist: {len(tickers)} tickers sampled (seed={wl_seed}, power={args.power}, min_cap={args.min_market_cap:.0f})")
    broker = AlpacaBroker(dry_run=not args.live)
    if args.garch_fhs_events:
        source = CompositeEventSource(
            YFinanceEarningsSource(),
            FomcCalendarSource(),
            ManualCalendarSource(MANUAL_EVENTS_CSV),
        )
        factory = GarchFhsFactory(source, event_window=args.event_window)
    elif args.garch_fhs:
        factory = garch_fhs_factory
    elif args.no_events:
        factory = bootstrap_factory
    else:
        factory = _default_forecaster_factory(event_window=args.event_window)
    result = run_daily(
        tickers, broker,
        bankroll=args.bankroll, target_dte=args.target_dte,
        kelly_fraction=args.kelly_fraction, max_fraction=args.max_fraction,
        n_paths=args.n_paths, seed=args.seed,
        forecaster_factory=factory,
        sell_threshold=args.sell_threshold,
        skip_manage=args.no_manage,
        max_workers=args.workers,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()
