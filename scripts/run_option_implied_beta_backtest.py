#!/usr/bin/env python3
"""
run_option_implied_beta_backtest.py — gate the Option-Implied Beta forecaster.

Runs the point-in-time rolling log-score backtest for the Option-Implied Beta
forecaster (historical SPY/IWM EOD option PDFs → Beta map + idiosyncratic) and the
two incumbents — **bootstrap** (leaderboard baseline) and **plain GARCH-FHS** (the
strongest non-event forecaster, +0.137 vs bootstrap) — on the SAME tickers/dates,
then reports the paired log-score lift + t-stat + win rate vs each (the standing
decision criterion, CLAUDE.md).

NOTE the GARCH-FHS arm is PLAIN (no event conditioning) — the production default is
GarchFhsFactory (event-conditioned, +0.224), which would require per-ticker earnings
calendars for 100 names. Plain FHS is the clean apples-to-apples vol comparison; the
event-conditioned default sits further ahead still.

Network-backed and slow: index PDFs are fetched from Alpaca's historical option
chains and cached per (index, run_date) — so they cost the same whether you run 3
tickers or 100. Use a date window safely in the PAST (contracts expired; after
Alpaca's Feb-2024 option-history start). The stock history starts earlier than the
holdout so the 252-day beta window is well-populated.

Usage:
    python scripts/run_option_implied_beta_backtest.py --sample 100 \
        --start 2023-01-01 --end 2025-05-01 --horizon 21 --holdout 63
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

import numpy as np
from scipy import stats

from datetime import timedelta

from options_trader.config import DEFAULT_RISK_FREE_RATE
from options_trader.data.history import get_history
from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.composite import CompositeEventSource
from options_trader.data.events.fomc_source import FomcCalendarSource
from options_trader.data.events.yfinance_source import YFinanceEarningsSource
from options_trader.backtest.log_score import (
    rolling_log_score_backtest,
    rolling_log_score_backtest_conditioned,
    rolling_log_score_backtest_garch_fhs_event,
)
from options_trader.backtest.option_implied_beta_backtest import (
    HistoricalEodPdf,
    _spot_lookup_from_ts,
    rolling_log_score_backtest_option_implied_beta,
    rolling_log_score_backtest_option_implied_beta_event,
)
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.garch_fhs_forecaster import GarchFhsForecaster
from options_trader.universe.sampler import sample_tickers


logger = logging.getLogger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _paired_report(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> str:
    """Paired comparison of two equal-length, date-aligned log-score arrays."""
    d = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    win = float((d > 0).mean())
    verdict = "ACCEPT" if (d.mean() > 0 and p_val < 0.05) else "do not accept"
    return (
        f"  {name_a} vs {name_b}: diff={d.mean():+.4f}  "
        f"t={t_stat:.3f}  p={p_val:.4g}  win={win:.0%}  -> {verdict}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="explicit tickers; if omitted, sample from the universe")
    ap.add_argument("--sample", type=int, default=100, help="number of tickers to sample")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--power", type=float, default=0.5,
                    help="cap-weight exponent for sampling (0.5 = ∝√cap, diverse)")
    ap.add_argument("--start", type=_parse_date, default=date(2023, 1, 1))
    ap.add_argument("--end", type=_parse_date, default=date(2025, 5, 1))
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--holdout", type=int, default=63)
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta-window", type=int, default=252)
    ap.add_argument("--events", action="store_true",
                    help="event-conditioned three-way: event-bootstrap vs event-GARCH-FHS vs "
                         "event-OIB (per-ticker yfinance earnings + FOMC calendar; slow)")
    ap.add_argument("--out", default="output/oib_3way_backtest.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    tickers = args.tickers or sample_tickers(
        args.sample, seed=args.sample_seed, power=args.power, exclude=("SPY", "IWM"))
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Window [{args.start} .. {args.end}]  h={args.horizon}  holdout={args.holdout}  "
          f"n_paths={args.n_paths}")

    print("Fetching index histories (SPY, IWM) ...", flush=True)
    spy = get_history("SPY", args.start, args.end)
    iwm = get_history("IWM", args.start, args.end)
    index_pdf = HistoricalEodPdf(_spot_lookup_from_ts(spy, iwm), risk_free_rate=DEFAULT_RISK_FREE_RATE)

    event_source = None
    if args.events:
        print("Event mode: building per-ticker earnings + FOMC calendars "
              "(event-bootstrap / event-GARCH-FHS / event-OIB).", flush=True)
        event_source = CompositeEventSource(YFinanceEarningsSource(), FomcCalendarSource())

    # Date-aligned, pooled log scores across tickers for each arm.
    pooled = {"oib": [], "boot": [], "garch": []}
    per_ticker = []
    n_fail = 0

    for i, tkr in enumerate(tickers, 1):
        try:
            ts = get_history(tkr, args.start, args.end)
            if args.events:
                cal = EventCalendar.from_source(
                    event_source, tkr, args.start, args.end + timedelta(days=60))
                oib = rolling_log_score_backtest_option_implied_beta_event(
                    ts, spy, iwm, index_pdf, cal, horizon_days=args.horizon,
                    holdout_days=args.holdout, n_paths=args.n_paths,
                    base_seed=args.seed, beta_window=args.beta_window)
                boot = rolling_log_score_backtest_conditioned(
                    ts, cal, horizon_days=args.horizon, holdout_days=args.holdout,
                    n_paths=args.n_paths, base_seed=args.seed)
                garch = rolling_log_score_backtest_garch_fhs_event(
                    ts, cal, horizon_days=args.horizon, holdout_days=args.holdout,
                    n_paths=args.n_paths, base_seed=args.seed)
            else:
                oib = rolling_log_score_backtest_option_implied_beta(
                    ts, spy, iwm, index_pdf, horizon_days=args.horizon,
                    holdout_days=args.holdout, n_paths=args.n_paths,
                    base_seed=args.seed, beta_window=args.beta_window)
                boot = rolling_log_score_backtest(
                    ts, lambda rd: BootstrapForecaster(rd, n_paths=args.n_paths, seed=args.seed),
                    horizon_days=args.horizon, holdout_days=args.holdout)
                garch = rolling_log_score_backtest(
                    ts, lambda rd: GarchFhsForecaster(rd, n_paths=args.n_paths, seed=args.seed),
                    horizon_days=args.horizon, holdout_days=args.holdout)

            boot_by = {e.forecast_date: e.log_score for e in boot.evaluations}
            garch_by = {e.forecast_date: e.log_score for e in garch.evaluations}
            rows = [(e.log_score, boot_by[e.forecast_date], garch_by[e.forecast_date])
                    for e in oib.evaluations
                    if e.forecast_date in boot_by and e.forecast_date in garch_by]
            if not rows:
                print(f"  [{i}/{len(tickers)}] {tkr}: no paired evals "
                      f"(skipped={oib.config['n_skipped']})", flush=True)
                continue
            o, b, g = (np.array(x) for x in zip(*rows))
            pooled["oib"].extend(o); pooled["boot"].extend(b); pooled["garch"].extend(g)
            per_ticker.append((tkr, len(rows), oib.config["n_skipped"],
                               o.mean(), b.mean(), g.mean()))
            print(f"  [{i}/{len(tickers)}] {tkr}: n={len(rows)} skip={oib.config['n_skipped']} "
                  f"oib={o.mean():.3f} boot={b.mean():.3f} garch={g.mean():.3f} "
                  f"(oib-boot={o.mean()-b.mean():+.3f} oib-garch={o.mean()-g.mean():+.3f})",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 — isolate per-ticker failures
            n_fail += 1
            print(f"  [{i}/{len(tickers)}] {tkr}: FAILED ({type(exc).__name__}: {exc})", flush=True)

    if not pooled["oib"]:
        print("\nNo paired evaluations across any ticker — cannot compare.")
        return

    oib = np.array(pooled["oib"]); boot = np.array(pooled["boot"]); garch = np.array(pooled["garch"])
    oib_wins_boot = sum(1 for _, _, _, o, b, _ in per_ticker if o > b)
    oib_wins_garch = sum(1 for _, _, _, o, _, g in per_ticker if o > g)
    mode = "EVENT-CONDITIONED" if args.events else "PLAIN (no events)"
    boot_name = "event_bootstrap" if args.events else "bootstrap"
    garch_name = "event_garch_fhs" if args.events else "garch_fhs (plain)"
    sanity = "~+0.224 expected" if args.events else "~+0.137 expected"

    lines = [
        "=" * 70,
        f"POOLED [{mode}]  n={len(oib)}  tickers={len(per_ticker)} (failed={n_fail})  "
        f"horizon={args.horizon}  holdout={args.holdout}  n_paths={args.n_paths}",
        f"  mean log score  option_implied_beta : {oib.mean():.4f}",
        f"  mean log score  {boot_name:<18}: {boot.mean():.4f}",
        f"  mean log score  {garch_name:<18}: {garch.mean():.4f}",
        "",
        _paired_report("oib  ", oib, boot_name, boot),
        _paired_report("oib  ", oib, garch_name, garch),
        _paired_report(garch_name, garch, boot_name, boot) + f"   (sanity: {sanity})",
        "",
        f"  per-ticker oib beats bootstrap : {oib_wins_boot}/{len(per_ticker)}",
        f"  per-ticker oib beats garch_fhs : {oib_wins_garch}/{len(per_ticker)}",
        "=" * 70,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    with open(args.out, "w") as fh:
        fh.write(f"# Option-Implied Beta 3-way backtest\n\n")
        fh.write(f"Window {args.start}..{args.end}, h={args.horizon}, holdout={args.holdout}, "
                 f"n_paths={args.n_paths}, sample={args.sample} power={args.power}\n\n")
        fh.write("## Per ticker (oib / boot / garch mean log score)\n\n")
        for tkr, n, skip, o, b, g in sorted(per_ticker, key=lambda r: r[3] - r[4], reverse=True):
            fh.write(f"- {tkr}: n={n} skip={skip} oib={o:.3f} boot={b:.3f} garch={g:.3f} "
                     f"(oib-boot={o-b:+.3f}, oib-garch={o-g:+.3f})\n")
        fh.write("\n## Pooled\n\n```\n" + report + "\n```\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
