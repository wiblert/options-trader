#!/usr/bin/env python3
"""
run_option_implied_backtest.py — is our forecast more accurate than the market price?

Scores the **option-implied (own-options) benchmark** forecaster — a ticker's OWN
risk-neutral PDF re-anchored to spot — head-to-head against the **production default**
(event-conditioned bootstrap) on the rolling log-score backtest, over the SAME
tickers/dates, then reports the paired log-score difference + t-stat + win rate.

INTERPRETATION
    option_implied = the option market's own implied view of S_T (the prices for sale).
    The reported difference is (option_implied − event_bootstrap):
      * NEGATIVE  -> our production forecast scores HIGHER than the market's own price
                     => we are more accurate than the prices for sale (predictive edge).
      * ~0 / positive -> the market price already encodes what we know (no edge).

⚠️ The option-implied arm is a BENCHMARK, never a production default — it reproduces the
prices already for sale (circular to trade against). See
forecast/option_implied_forecaster.py.

Network-backed and slow: each run_date fetches the ticker's historical EOD option chain
from Alpaca (cached per (ticker, run_date)). Use a date window safely in the PAST
(contracts expired; after Alpaca's Feb-2024 option-history start). The stock history
should start well before the holdout so the production forecaster trains on enough data.

Usage:
    python scripts/run_option_implied_backtest.py --tickers META AAPL \
        --start 2024-06-01 --end 2025-05-01 --horizon 21 --holdout 63
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

import numpy as np
from scipy import stats

from options_trader.data.history import get_history
from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.composite import CompositeEventSource
from options_trader.data.events.fomc_source import FomcCalendarSource
from options_trader.data.events.yfinance_source import YFinanceEarningsSource
from options_trader.backtest.log_score import rolling_log_score_backtest_conditioned
from options_trader.backtest.option_implied_backtest import (
    HistoricalTickerPdf,
    _spot_lookup_from_ts,
    rolling_log_score_backtest_option_implied,
)
from options_trader.universe.sampler import sample_tickers


logger = logging.getLogger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _paired_report(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> str:
    """Paired comparison of two equal-length, date-aligned log-score arrays."""
    d = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    win = float((d > 0).mean())
    return (
        f"  {name_a} vs {name_b}: diff={d.mean():+.4f}  "
        f"t={t_stat:.3f}  p={p_val:.4g}  win={win:.0%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="explicit tickers; if omitted, sample from the universe")
    ap.add_argument("--sample", type=int, default=10, help="number of tickers to sample")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--power", type=float, default=0.5,
                    help="cap-weight exponent for sampling (0.5 = ∝√cap, diverse)")
    ap.add_argument("--start", type=_parse_date, default=date(2024, 6, 1))
    ap.add_argument("--end", type=_parse_date, default=date(2025, 5, 1))
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--holdout", type=int, default=63)
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output/option_implied_backtest.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    tickers = args.tickers or sample_tickers(
        args.sample, seed=args.sample_seed, power=args.power, exclude=("SPY", "IWM"))
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Window [{args.start} .. {args.end}]  h={args.horizon}  holdout={args.holdout}  "
          f"n_paths={args.n_paths}")
    print("Incumbent = event_bootstrap (production default). "
          "Negative diff => our forecast beats the market price.")

    event_source = CompositeEventSource(YFinanceEarningsSource(), FomcCalendarSource())

    # Date-aligned, pooled log scores across tickers for each arm.
    pooled = {"oi": [], "boot": []}
    per_ticker = []
    n_fail = 0

    for i, tkr in enumerate(tickers, 1):
        try:
            ts = get_history(tkr, args.start, args.end)
            ticker_pdf = HistoricalTickerPdf(_spot_lookup_from_ts(ts), risk_free_rate=0.04)

            oi = rolling_log_score_backtest_option_implied(
                ts, ticker_pdf, horizon_days=args.horizon, holdout_days=args.holdout,
                n_paths=args.n_paths, base_seed=args.seed)

            cal = EventCalendar.from_source(
                event_source, tkr, args.start, args.end + timedelta(days=60))
            boot = rolling_log_score_backtest_conditioned(
                ts, cal, horizon_days=args.horizon, holdout_days=args.holdout,
                n_paths=args.n_paths, base_seed=args.seed)

            boot_by = {e.forecast_date: e.log_score for e in boot.evaluations}
            rows = [(e.log_score, boot_by[e.forecast_date])
                    for e in oi.evaluations if e.forecast_date in boot_by]
            if not rows:
                print(f"  [{i}/{len(tickers)}] {tkr}: no paired evals "
                      f"(skipped={oi.config['n_skipped']})", flush=True)
                continue
            o, b = (np.array(x) for x in zip(*rows))
            pooled["oi"].extend(o); pooled["boot"].extend(b)
            per_ticker.append((tkr, len(rows), oi.config["n_skipped"], o.mean(), b.mean()))
            print(f"  [{i}/{len(tickers)}] {tkr}: n={len(rows)} skip={oi.config['n_skipped']} "
                  f"option_implied={o.mean():.3f} event_bootstrap={b.mean():.3f} "
                  f"(diff={o.mean()-b.mean():+.3f})", flush=True)
        except Exception as exc:  # noqa: BLE001 — isolate per-ticker failures
            n_fail += 1
            print(f"  [{i}/{len(tickers)}] {tkr}: FAILED ({type(exc).__name__}: {exc})", flush=True)

    if not pooled["oi"]:
        print("\nNo paired evaluations across any ticker — cannot compare.")
        return

    oi = np.array(pooled["oi"]); boot = np.array(pooled["boot"])
    oi_wins = sum(1 for _, _, _, o, b in per_ticker if o > b)

    lines = [
        "=" * 70,
        f"POOLED  n={len(oi)}  tickers={len(per_ticker)} (failed={n_fail})  "
        f"horizon={args.horizon}  holdout={args.holdout}  n_paths={args.n_paths}",
        f"  mean log score  option_implied (market price) : {oi.mean():.4f}",
        f"  mean log score  event_bootstrap (production)  : {boot.mean():.4f}",
        "",
        _paired_report("option_implied", oi, "event_bootstrap", boot),
        "",
        "  diff = (option_implied - event_bootstrap):",
        "    NEGATIVE -> production forecast more accurate than the market price (EDGE)",
        "    >= 0     -> market price already encodes our information (no edge)",
        f"  per-ticker option_implied beats event_bootstrap : {oi_wins}/{len(per_ticker)}",
        "=" * 70,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    with open(args.out, "w") as fh:
        fh.write("# Option-Implied (own-options) benchmark vs production (event-bootstrap)\n\n")
        fh.write(f"Window {args.start}..{args.end}, h={args.horizon}, holdout={args.holdout}, "
                 f"n_paths={args.n_paths}\n\n")
        fh.write("option_implied = the market's own implied view (the prices for sale); "
                 "a NEGATIVE diff means our production forecast is more accurate "
                 "(predictive edge).\n\n")
        fh.write("## Per ticker (option_implied / event_bootstrap mean log score)\n\n")
        for tkr, n, skip, o, b in sorted(per_ticker, key=lambda r: r[3] - r[4]):
            fh.write(f"- {tkr}: n={n} skip={skip} option_implied={o:.3f} "
                     f"event_bootstrap={b:.3f} (diff={o-b:+.3f})\n")
        fh.write("\n## Pooled\n\n```\n" + report + "\n```\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
