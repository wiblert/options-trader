"""
compare_forecasters.py — side-by-side comparison of GARCH-FHS vs event-conditioned
bootstrap on today's watchlist.

For each ticker the script fetches history and the option chain ONCE, then runs both
forecasters on the same data.  This halves the API load vs calling plan_ticker twice.

Output: a table of contracts where the two forecasters disagree (one BUY, one not) or
both agree to BUY (showing the edge/sizing delta).  BOTH_NOBUY rows are suppressed.

Usage
-----
    python scripts/compare_forecasters.py
    python scripts/compare_forecasters.py --tickers AAPL,GOOG,MSFT
    python scripts/compare_forecasters.py --n-tickers 20 --target-dte 21

Columns
-------
    ticker  spot  contract  type  ask  fhs_edge  evt_edge  Δedge  fhs_N  evt_N  verdict

Verdict values
--------------
    FHS_ONLY   — GARCH-FHS says BUY, event-bootstrap does not
    EVENT_ONLY — event-bootstrap says BUY, GARCH-FHS does not
    BOTH_BUY   — both say BUY (Δedge = fhs_edge − evt_edge shown)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
from options_trader.config import DEFAULT_RISK_FREE_RATE
from options_trader.data.history import get_history, HistoryError
from options_trader.data.options_chain import get_option_chain, OptionsChainError
from options_trader.data.spot_anchor import live_else_close_anchor
from options_trader.forecast.factories import (
    ForecastContext,
    garch_fhs_factory,
)
from options_trader.forecast.return_distribution import ReturnDistribution
from options_trader.run_daily import (
    _select_expiry,
    _default_forecaster_factory,
    _daily_watchlist_seed,
    _default_watchlist,
    DEFAULT_TARGET_DTE,
    DEFAULT_N_PATHS,
    DEFAULT_N_TICKERS,
    DEFAULT_WATCHLIST_POWER,
)
from options_trader.sizing.kelly import KellySizer, KellySizing, SizingStatus
from options_trader.valuation.option_valuer import (
    OptionValuation,
    OptionValuer,
    Recommendation,
)


logger = logging.getLogger(__name__)

DUMMY_BANKROLL = 100_000.0  # sizing reference; relative values matter, not absolutes


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class ContractResult:
    """One forecaster's view of a single contract."""
    valuation: Optional[OptionValuation]
    n_contracts: Optional[int]
    is_buy: bool


@dataclass
class ContractComparison:
    symbol: str
    option_type: str          # "call" / "put"
    strike: float
    ask: float
    fhs: ContractResult
    event: ContractResult

    @property
    def verdict(self) -> str:
        if self.fhs.is_buy and self.event.is_buy:
            return "BOTH_BUY"
        if self.fhs.is_buy:
            return "FHS_ONLY"
        if self.event.is_buy:
            return "EVENT_ONLY"
        return "BOTH_NOBUY"

    @property
    def fhs_edge(self) -> Optional[float]:
        return self.fhs.valuation.edge_buy if self.fhs.valuation else None

    @property
    def event_edge(self) -> Optional[float]:
        return self.event.valuation.edge_buy if self.event.valuation else None

    @property
    def delta_edge(self) -> Optional[float]:
        if self.fhs_edge is not None and self.event_edge is not None:
            return self.fhs_edge - self.event_edge
        return None


@dataclass
class TickerComparison:
    ticker: str
    spot: Optional[float] = None
    expiry: Optional[date] = None
    horizon: Optional[int] = None
    events_in_horizon: Optional[int] = None
    fhs_forecast_mean: Optional[float] = None
    event_forecast_mean: Optional[float] = None
    comparisons: list[ContractComparison] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def interesting(self) -> list[ContractComparison]:
        """All rows except BOTH_NOBUY."""
        return [c for c in self.comparisons if c.verdict != "BOTH_NOBUY"]


# ── core comparison logic ──────────────────────────────────────────────────────

def _score_contract(
    pdist,
    contract,
    spot: float,
    run_date: date,
    valuer: OptionValuer,
    sizer: KellySizer,
) -> ContractResult:
    """Value and size a contract under one price distribution."""
    v = valuer.value(pdist, contract, spot=spot, valuation_date=run_date)
    is_buy = v.recommendation == Recommendation.BUY
    if not is_buy:
        return ContractResult(valuation=v, n_contracts=None, is_buy=False)
    s = sizer.size(pdist, v, bankroll=DUMMY_BANKROLL)
    n = s.n_contracts if s.status == SizingStatus.SIZED else None
    return ContractResult(valuation=v, n_contracts=n, is_buy=is_buy)


def compare_ticker(
    ticker: str,
    *,
    run_date: date,
    valuer: OptionValuer,
    sizer: KellySizer,
    event_factory,
    target_dte: int = DEFAULT_TARGET_DTE,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = 42,
    strike_window: float = 0.08,
) -> TickerComparison:
    result = TickerComparison(ticker=ticker)
    try:
        ts = get_history(ticker)
        anchored = live_else_close_anchor(ticker, ts)
        spot = anchored.spot
        result.spot = spot

        expiry, horizon = _select_expiry(ticker, spot, target_dte, run_date)
        result.expiry, result.horizon = expiry, horizon
        if horizon < 1:
            result.error = f"expiry {expiry} is <1 trading day out"
            return result

        rd = ReturnDistribution(np.diff(np.log(ts.close)))
        ctx = ForecastContext(
            ticker=ticker, ts=ts, rd=rd, horizon=horizon,
            run_date=run_date, n_paths=n_paths, seed=seed,
        )

        fhs_forecaster = garch_fhs_factory(ctx)
        event_forecaster = event_factory(ctx)

        pdist_fhs = fhs_forecaster.forecast(horizon, spot)
        pdist_event = event_forecaster.forecast(horizon, spot)
        result.fhs_forecast_mean = float(pdist_fhs.mean())
        result.event_forecast_mean = float(pdist_event.mean())

        # count event days routed by the event forecaster
        schedule = getattr(event_forecaster, "event_schedule", None)
        result.events_in_horizon = sum(1 for s in schedule if s is not None) if schedule else 0

        chain = get_option_chain(
            ticker,
            expiration_gte=expiry, expiration_lte=expiry,
            strike_gte=spot * (1 - strike_window),
            strike_lte=spot * (1 + strike_window),
        )

        for c in chain:
            if c.ask is None:
                continue
            fhs_res = _score_contract(pdist_fhs, c, spot, run_date, valuer, sizer)
            event_res = _score_contract(pdist_event, c, spot, run_date, valuer, sizer)
            result.comparisons.append(ContractComparison(
                symbol=c.symbol,
                option_type=c.option_type.value,
                strike=c.strike,
                ask=c.ask,
                fhs=fhs_res,
                event=event_res,
            ))
    except (HistoryError, OptionsChainError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        logger.exception("Unexpected error comparing %s", ticker)
        result.error = f"{type(exc).__name__}: {exc}"
    return result


# ── output ─────────────────────────────────────────────────────────────────────

def _print_results(results: list[TickerComparison], run_date: date) -> None:
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    print(f"\n=== Forecaster comparison: GARCH-FHS vs Event-Bootstrap | {run_date} ===\n")

    rows = []
    for r in sorted(results, key=lambda x: x.ticker):
        if r.error:
            rows.append([r.ticker, "—", "—", "—", "—", "—", "—", "—", "—", "—",
                         f"ERROR: {r.error[:40]}"])
            continue
        ev = str(r.events_in_horizon) if r.events_in_horizon else ""
        if not r.interesting:
            # both agree nothing to buy — show one summary row
            rows.append([r.ticker, f"{r.spot:.2f}", str(r.expiry), ev,
                         "—", "—", "—", "—", "—", "—", "no-BUY (both agree)"])
            continue
        for c in sorted(r.interesting, key=lambda x: (x.option_type, x.strike)):
            fhs_edge = f"{c.fhs_edge:+.2f}" if c.fhs_edge is not None else "—"
            evt_edge = f"{c.event_edge:+.2f}" if c.event_edge is not None else "—"
            delta = f"{c.delta_edge:+.2f}" if c.delta_edge is not None else "—"
            fhs_n = str(c.fhs.n_contracts) if c.fhs.n_contracts is not None else "—"
            evt_n = str(c.event.n_contracts) if c.event.n_contracts is not None else "—"
            rows.append([
                r.ticker, f"{r.spot:.2f}", str(r.expiry), ev,
                f"{c.option_type} {c.strike:.0f}", f"{c.ask:.2f}",
                fhs_edge, evt_edge, delta, fhs_n, evt_n,
                c.verdict,
            ])

    headers = ["ticker", "spot", "expiry", "ev", "contract", "ask",
               "fhs_edge", "evt_edge", "Δedge", "fhs_N", "evt_N", "verdict"]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(x) for x in row))

    # summary counts
    all_cmp = [c for r in results if not r.error for c in r.comparisons]
    fhs_only = sum(1 for c in all_cmp if c.verdict == "FHS_ONLY")
    evt_only = sum(1 for c in all_cmp if c.verdict == "EVENT_ONLY")
    both_buy = sum(1 for c in all_cmp if c.verdict == "BOTH_BUY")
    n_err = sum(1 for r in results if r.error)
    print(
        f"\n{len(results)} tickers | {fhs_only} FHS_ONLY | {evt_only} EVENT_ONLY | "
        f"{both_buy} BOTH_BUY | {n_err} errors"
    )

    # forecast-mean comparison for tickers where they diverge materially
    mean_diffs = [
        (r.ticker, r.fhs_forecast_mean, r.event_forecast_mean,
         100 * (r.fhs_forecast_mean - r.event_forecast_mean) / r.spot)
        for r in results
        if r.spot and r.fhs_forecast_mean and r.event_forecast_mean
        and abs(r.fhs_forecast_mean - r.event_forecast_mean) / r.spot > 0.005
    ]
    if mean_diffs:
        print("\nForecast-mean divergence >0.5% of spot (FHS vs Event):")
        mean_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        if tabulate:
            print(tabulate(
                [[t, f"{fhs:.2f}", f"{ev:.2f}", f"{d:+.2f}%"]
                 for t, fhs, ev, d in mean_diffs],
                headers=["ticker", "fhs_mean", "evt_mean", "Δ%"],
                tablefmt="github",
            ))
        else:
            for t, fhs, ev, d in mean_diffs:
                print(f"  {t}: fhs={fhs:.2f}  evt={ev:.2f}  Δ={d:+.2f}%")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Compare GARCH-FHS vs event-bootstrap on today's watchlist."
    )
    ap.add_argument("--tickers", type=str, default=None,
                    help="comma-separated tickers; default = universe sample")
    ap.add_argument("--n-tickers", type=int, default=DEFAULT_N_TICKERS)
    ap.add_argument("--target-dte", type=int, default=DEFAULT_TARGET_DTE)
    ap.add_argument("--n-paths", type=int, default=DEFAULT_N_PATHS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--watchlist-seed", type=int, default=None)
    ap.add_argument("--power", type=float, default=DEFAULT_WATCHLIST_POWER)
    ap.add_argument("--event-window", type=int, default=0)
    args = ap.parse_args()

    run_date = date.today()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        wl_seed = args.watchlist_seed if args.watchlist_seed is not None \
            else _daily_watchlist_seed(run_date)
        tickers = _default_watchlist(args.n_tickers, wl_seed, args.power)
        print(f"watchlist: {len(tickers)} tickers (seed={wl_seed}, power={args.power})")

    event_factory = _default_forecaster_factory(event_window=args.event_window)
    valuer = OptionValuer(risk_free_rate=DEFAULT_RISK_FREE_RATE)
    sizer = KellySizer(kelly_fraction=0.25, max_fraction=0.05)

    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker} ...", end="\r", flush=True)
        results.append(compare_ticker(
            ticker, run_date=run_date, valuer=valuer, sizer=sizer,
            event_factory=event_factory, target_dte=args.target_dte,
            n_paths=args.n_paths, seed=args.seed,
        ))
    print(" " * 40, end="\r")  # clear progress line

    _print_results(results, run_date)


if __name__ == "__main__":
    main()
