#!/usr/bin/env python3
"""
run_standardized_test.py — CLI for the standardized forecaster benchmark.

Builds the test set (META + N cap-weighted sampled tickers from a frozen S&P 500
snapshot), runs Block/Bootstrap/Gaussian through the rolling log-score backtest,
and writes a markdown report + per-ticker CSV to output/.

Examples
    # Standard run: META + 19 cap-weighted tickers, seed 42
    python scripts/run_standardized_test.py

    # Re-snapshot the universe first (deliberate refresh), then run
    python scripts/run_standardized_test.py --refresh-universe

    # Cheaper smoke run
    python scripts/run_standardized_test.py --n-sampled 3 --n-paths 2000 --holdout 30
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from options_trader.backtest.anchor import MA_BLEND_ANCHOR_FNS
from options_trader.backtest.standardized_test import (
    DAMPENING_BASELINE_ARM,
    DEFAULT_ANCHOR_HEADLINE,
    DEFAULT_BASE_SEED,
    DEFAULT_DAMPENING_KAPPAS,
    DEFAULT_FORECASTERS,
    DEFAULT_GATE_TAUS,
    EVENT_FORECASTER_NAME,
    GARCH_FHS_EVENT_FORECASTER_NAME,
    GATE_ALWAYS_FHS_ARM,
    GATE_BOOTSTRAP_ARM,
    MA_BLEND_ANCHOR_HEADLINE,
    best_dampening_arm,
    best_gate_arm,
    run_anchor_experiment,
    run_drift_dampening_experiment,
    run_vol_gate_experiment,
    run_standardized_test,
)
from options_trader.data.events.composite import CompositeEventSource
from options_trader.data.events.fomc_source import FomcCalendarSource
from options_trader.data.events.manual_calendar import ManualCalendarSource
from options_trader.data.events.yfinance_source import YFinanceEarningsSource
from options_trader.universe.sampler import sample_tickers
from options_trader.universe.snapshot import latest_snapshot_path, refresh_snapshot


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
MANUAL_EVENTS_CSV = REPO_ROOT / "config" / "manual_events.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixed", nargs="*", default=["META"],
                   help="tickers always in the test set (default: META)")
    p.add_argument("--n-sampled", type=int, default=19,
                   help="number of cap-weighted tickers to draw (default: 19)")
    p.add_argument("--sampler-seed", type=int, default=42,
                   help="seed for the cap-weighted draw (default: 42)")
    p.add_argument("--power", type=float, default=1.0,
                   help="cap-weight exponent: 1.0=cap-weighted, 0.0=uniform (default: 1.0)")
    p.add_argument("--horizon", type=int, default=5, help="forecast horizon in days")
    p.add_argument("--holdout", type=int, default=63, help="out-of-sample days")
    p.add_argument("--n-paths", type=int, default=10_000, help="MC paths per forecast")
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="forecaster MC base seed")
    p.add_argument("--decay-lambda", type=float, default=0.99, help="return-distribution decay λ")
    p.add_argument("--refresh-universe", action="store_true",
                   help="re-snapshot the S&P 500 universe before sampling (network)")
    p.add_argument("--snapshot", type=Path, default=None,
                   help="explicit snapshot CSV path (default: latest committed)")
    p.add_argument("--events", action="store_true",
                   help="add the event_bootstrap forecaster (yfinance earnings + manual calendar)")
    p.add_argument("--garch-fhs", action="store_true",
                   help="add the GARCH(1,1)-filtered FHS forecaster; headline = garch_fhs vs bootstrap")
    p.add_argument("--garch-fhs-events", action="store_true",
                   help="add GARCH-FHS + event-conditioned forecaster; requires event data")
    p.add_argument("--event-window", type=int, default=0,
                   help="include +/-N returns around each event (default: 0)")
    p.add_argument("--anchor-experiment", action="store_true",
                   help="instead of comparing forecasters, hold Bootstrap fixed and compare "
                        "spot anchors (see --anchor-set)")
    p.add_argument("--anchor-set", choices=["intraday", "ma-blend"], default="intraday",
                   help="which anchor comparison to run with --anchor-experiment: "
                        "'intraday' = intraday_random vs stale_close (validated S9); "
                        "'ma-blend' = ma7_blend / ma20_blend / ma7_only vs intraday_random "
                        "(Hypothesis 1: smooth the fresh price toward a trailing MA)")
    p.add_argument("--drift-dampening-experiment", action="store_true",
                   help="sweep the drift-dampening fraction κ on the bootstrap "
                        "(terminal mean shrunk toward spot); fits κ by pooled log score")
    p.add_argument("--kappas", type=float, nargs="*", default=None,
                   help="κ grid for --drift-dampening-experiment "
                        f"(default: {list(DEFAULT_DAMPENING_KAPPAS)}); 0.0 is always included")
    p.add_argument("--vol-gate-experiment", action="store_true",
                   help="sweep the FHS vol-gate threshold τ (use FHS when σ_next/σ̄ ≥ τ, "
                        "else bootstrap); reports best τ vs both always-bootstrap and always-FHS")
    p.add_argument("--taus", type=float, nargs="*", default=None,
                   help="τ grid for --vol-gate-experiment "
                        f"(default: {[t for t in DEFAULT_GATE_TAUS if t != float('inf')] + ['inf']}); "
                        "0.0 (always FHS) and inf (always bootstrap) are always included")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.refresh_universe:
        print("Refreshing S&P 500 snapshot (this takes a few minutes)…")
        refresh_snapshot()

    snapshot_path = args.snapshot or latest_snapshot_path()
    print(f"Universe snapshot: {snapshot_path.name}")

    sampled = sample_tickers(
        n=args.n_sampled,
        seed=args.sampler_seed,
        snapshot_path=snapshot_path,
        power=args.power,
        exclude=args.fixed,
    )
    tickers = list(dict.fromkeys([*args.fixed, *sampled]))  # fixed first, dedup
    print(f"Test set ({len(tickers)}): {', '.join(tickers)}\n")

    config_extra = {
        "fixed_tickers": args.fixed,
        "n_sampled": args.n_sampled,
        "sampler_seed": args.sampler_seed,
        "power": args.power,
        "snapshot_path": str(snapshot_path.name),
    }

    if args.anchor_experiment:
        ma_blend = args.anchor_set == "ma-blend"
        anchor_fns = MA_BLEND_ANCHOR_FNS if ma_blend else None
        headline = MA_BLEND_ANCHOR_HEADLINE if ma_blend else DEFAULT_ANCHOR_HEADLINE
        result = run_anchor_experiment(
            tickers=tickers,
            horizon_days=args.horizon,
            holdout_days=args.holdout,
            n_paths=args.n_paths,
            base_seed=args.base_seed,
            decay_lambda=args.decay_lambda,
            anchor_fns=anchor_fns,
            config_extra=config_extra,
        )
        report = result.to_report(headline=headline)
        print("\n" + report)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = f"{date.today().isoformat()}_anchor-{args.anchor_set}_h{args.horizon}_hold{args.holdout}"
        report_path = OUTPUT_DIR / f"anchor_experiment_{stamp}.md"
        csv_path = OUTPUT_DIR / f"anchor_experiment_{stamp}.csv"
        report_path.write_text(report)
        result.summary_table().to_csv(csv_path)
        print(f"\nWrote {report_path}\n      {csv_path}")
        return

    if args.drift_dampening_experiment:
        kappas = tuple(args.kappas) if args.kappas else DEFAULT_DAMPENING_KAPPAS
        result = run_drift_dampening_experiment(
            tickers=tickers,
            kappas=kappas,
            horizon_days=args.horizon,
            holdout_days=args.holdout,
            n_paths=args.n_paths,
            base_seed=args.base_seed,
            decay_lambda=args.decay_lambda,
            config_extra=config_extra,
        )
        best_arm, best_score = best_dampening_arm(result)
        report = result.to_report(headline=(best_arm, DAMPENING_BASELINE_ARM))
        print("\n" + report)
        print(f"\nFitted κ: best arm = {best_arm} (pooled mean log score {best_score:+.4f}); "
              f"baseline = {DAMPENING_BASELINE_ARM}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = f"{date.today().isoformat()}_dampening_h{args.horizon}_hold{args.holdout}"
        report_path = OUTPUT_DIR / f"dampening_experiment_{stamp}.md"
        csv_path = OUTPUT_DIR / f"dampening_experiment_{stamp}.csv"
        report_path.write_text(report)
        result.summary_table().to_csv(csv_path)
        print(f"\nWrote {report_path}\n      {csv_path}")
        return

    if args.vol_gate_experiment:
        taus = tuple(args.taus) if args.taus else DEFAULT_GATE_TAUS
        result = run_vol_gate_experiment(
            tickers=tickers, taus=taus,
            horizon_days=args.horizon, holdout_days=args.holdout,
            n_paths=args.n_paths, base_seed=args.base_seed,
            decay_lambda=args.decay_lambda, config_extra=config_extra,
        )
        best_arm, best_score = best_gate_arm(result)
        report = result.to_report(headline=(best_arm, GATE_BOOTSTRAP_ARM))
        print("\n" + report)
        # Two reference comparisons: gating vs incumbent bootstrap, and vs plain FHS.
        vs_boot = result.paired_comparison(best_arm, GATE_BOOTSTRAP_ARM)
        vs_fhs = result.paired_comparison(best_arm, GATE_ALWAYS_FHS_ARM)
        print(f"\nFitted τ: best arm = {best_arm} (pooled mean log score {best_score:+.4f})")
        print(f"  vs always-bootstrap ({GATE_BOOTSTRAP_ARM}): {vs_boot['mean_diff']:+.4f} "
              f"(t={vs_boot['t_stat']:.2f}, p={vs_boot['p_value']:.3g}, "
              f"{vs_boot['win_rate_by_ticker']:.0%} win)")
        print(f"  vs always-FHS      ({GATE_ALWAYS_FHS_ARM}): {vs_fhs['mean_diff']:+.4f} "
              f"(t={vs_fhs['t_stat']:.2f}, p={vs_fhs['p_value']:.3g}, "
              f"{vs_fhs['win_rate_by_ticker']:.0%} win)")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = f"{date.today().isoformat()}_volgate_h{args.horizon}_hold{args.holdout}"
        (OUTPUT_DIR / f"vol_gate_experiment_{stamp}.md").write_text(report)
        result.summary_table().to_csv(OUTPUT_DIR / f"vol_gate_experiment_{stamp}.csv")
        print(f"\nWrote {OUTPUT_DIR / f'vol_gate_experiment_{stamp}.md'}")
        return

    forecasters = DEFAULT_FORECASTERS
    if args.garch_fhs:
        forecasters = (*forecasters, "garch_fhs")
        print("GARCH-FHS enabled: garch_fhs forecaster added (headline vs bootstrap)\n")
    event_source = None
    if args.events or args.garch_fhs_events:
        event_source = CompositeEventSource(
            YFinanceEarningsSource(),
            FomcCalendarSource(),
            ManualCalendarSource(MANUAL_EVENTS_CSV),
        )
        print("Events enabled: yfinance earnings + FOMC calendar + manual calendar\n")
    if args.garch_fhs_events:
        forecasters = (*forecasters, GARCH_FHS_EVENT_FORECASTER_NAME)
        print("GARCH-FHS-Events enabled: GARCH-FHS + event routing added\n")
    if args.events:
        forecasters = (*forecasters, EVENT_FORECASTER_NAME)
        print("Event-bootstrap enabled: event_bootstrap forecaster added\n")

    result = run_standardized_test(
        tickers=tickers,
        forecasters=forecasters,
        horizon_days=args.horizon,
        holdout_days=args.holdout,
        n_paths=args.n_paths,
        base_seed=args.base_seed,
        decay_lambda=args.decay_lambda,
        event_source=event_source,
        event_window=args.event_window,
        config_extra=config_extra,
    )

    # Headline: garch_fhs_events > events > garch_fhs > default block-vs-bootstrap.
    if args.garch_fhs_events:
        headline = (GARCH_FHS_EVENT_FORECASTER_NAME, "bootstrap")
    elif args.events:
        headline = (EVENT_FORECASTER_NAME, "bootstrap")
    elif args.garch_fhs:
        headline = ("garch_fhs", "bootstrap")
    else:
        headline = ("block_bootstrap", "bootstrap")
    report = result.to_report(headline=headline)
    print("\n" + report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Stamp horizon + holdout into the filename so runs at different settings
    # don't clobber each other.
    stamp = f"{date.today().isoformat()}_h{args.horizon}_hold{args.holdout}"
    report_path = OUTPUT_DIR / f"standardized_test_{stamp}.md"
    csv_path = OUTPUT_DIR / f"standardized_test_{stamp}.csv"
    report_path.write_text(report)
    result.summary_table().to_csv(csv_path)
    print(f"\nWrote {report_path}\n      {csv_path}")


if __name__ == "__main__":
    main()
