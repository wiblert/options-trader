#!/usr/bin/env python3
"""
run_garch_vs_bootstrap_check.py — does event-GARCH-FHS earn its complexity?

Runs the canonical standardized set (META + 19 cap-weighted, seed 42) through the
project's `run_standardized_test`, with BOTH event forecasters added, then prints
the head-to-head the leaderboard never reports directly:

    event_garch_fhs  vs  event_bootstrap

plus each vs plain bootstrap (to reproduce the leaderboard: event_bootstrap ≈ +0.04,
garch_fhs_events ≈ +0.224). The question: is garch_fhs_events − event_bootstrap a
large, significant lift (GARCH's conditional-vol machinery earns its keep) or small
(the decay-weighted event-bootstrap already captures most of it)?

Recent window (get_history default ~3y) — the regime where garch_fhs_events was
validated, so it's a fair test of the incumbent (unlike the 2024-25 OIB windows).
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from options_trader.backtest.standardized_test import (
    DEFAULT_FORECASTERS,
    EVENT_FORECASTER_NAME,
    GARCH_FHS_EVENT_FORECASTER_NAME,
    run_standardized_test,
)
from options_trader.data.events.composite import CompositeEventSource
from options_trader.data.events.fomc_source import FomcCalendarSource
from options_trader.data.events.manual_calendar import ManualCalendarSource
from options_trader.data.events.yfinance_source import YFinanceEarningsSource
from options_trader.universe.sampler import sample_tickers
from options_trader.universe.snapshot import latest_snapshot_path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
MANUAL_EVENTS_CSV = REPO_ROOT / "config" / "manual_events.csv"


def _fmt(c: dict) -> str:
    verdict = "ACCEPT" if (c["mean_diff"] > 0 and c["p_value"] < 0.05) else "do not accept"
    return (f"{c['a']} vs {c['b']}: diff={c['mean_diff']:+.4f}  t={c['t_stat']:.2f}  "
            f"p={c['p_value']:.3g}  by-ticker win={c['win_rate_by_ticker']:.0%} "
            f"(n={c['n_pairs']}, {c['n_tickers']} tkrs)  -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sampled", type=int, default=19)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--holdout", type=int, default=63)
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    snap = latest_snapshot_path()
    sampled = sample_tickers(n=args.n_sampled, seed=42, snapshot_path=snap,
                             power=1.0, exclude=["META"])
    tickers = list(dict.fromkeys(["META", *sampled]))
    print(f"Standardized set ({len(tickers)}): {', '.join(tickers)}")
    print(f"h={args.horizon} holdout={args.holdout} n_paths={args.n_paths}  snapshot={snap.name}\n")

    event_source = CompositeEventSource(
        YFinanceEarningsSource(), FomcCalendarSource(), ManualCalendarSource(MANUAL_EVENTS_CSV))
    forecasters = (*DEFAULT_FORECASTERS, GARCH_FHS_EVENT_FORECASTER_NAME, EVENT_FORECASTER_NAME)

    result = run_standardized_test(
        tickers=tickers, forecasters=forecasters,
        horizon_days=args.horizon, holdout_days=args.holdout,
        n_paths=args.n_paths, event_source=event_source)

    print(result.to_report(headline=(GARCH_FHS_EVENT_FORECASTER_NAME, "bootstrap")))

    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD (does GARCH's complexity earn its keep?)")
    print("  " + _fmt(result.paired_comparison(GARCH_FHS_EVENT_FORECASTER_NAME, "bootstrap")))
    print("  " + _fmt(result.paired_comparison(EVENT_FORECASTER_NAME, "bootstrap")))
    print("  >>> " + _fmt(result.paired_comparison(
        GARCH_FHS_EVENT_FORECASTER_NAME, EVENT_FORECASTER_NAME)) + "   <<< the key one")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{date.today().isoformat()}_h{args.horizon}_hold{args.holdout}"
    out = OUTPUT_DIR / f"garch_vs_eventboot_{stamp}.md"
    out.write_text(result.to_report(headline=(GARCH_FHS_EVENT_FORECASTER_NAME, EVENT_FORECASTER_NAME)))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
