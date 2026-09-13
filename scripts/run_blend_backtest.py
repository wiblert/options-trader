#!/usr/bin/env python3
"""
run_blend_backtest.py — the 50/50 blend (event-bootstrap ⊕ event-OIB): comparison + tests.

Two modes:

  --mode vs-eventboot   (ITEMS 1 + 3a; default; ~40 names, 1yr)
      Runs the blend backtest per ticker (records event-bootstrap, event-OIB AND blend
      per-eval log scores in lockstep). Produces:
        * ITEM 1 chart  <prefix>_compare.png — where event-bootstrap vs OIB each win
          (overlaid log-score distributions, the per-eval (oib−eboot) histogram, the
          per-ticker mean (oib−eboot) bar, and the blend's (blend−eboot) histogram).
        * ITEM 3a report <prefix>_blend_vs_eventboot.md — paired blend−eboot (mean, t,
          p, win%, per-ticker wins); blend−oib for context.

  --mode vs-options     (ITEM 3b; ~20 names, 1yr)
      Runs the blend backtest AND the own-options benchmark per ticker, date-aligns,
      and reports paired blend − option_implied. (Own-options coverage is sparse for
      illiquid mid-caps — skips are counted/reported honestly.)

advantage convention everywhere: a − b, higher log score = more accurate.

Network-backed: SPY/IWM index PDFs are fetched once and cached per (index, run_date),
shared across all tickers. vs-options additionally fetches each ticker's own historical
chain per eval date (slow). Use a PAST window after Alpaca's Feb-2024 option-history
start; the stock history starts earlier so the 252-day beta window is populated.

Usage:
    python scripts/run_blend_backtest.py --mode vs-eventboot --sample 40 --sample-seed 11 \
        --start 2022-06-01 --end 2025-06-01 --horizon 21 --holdout 252 --n-paths 6000
    python scripts/run_blend_backtest.py --mode vs-options --sample 20 --sample-seed 23 ...
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from options_trader.config import DEFAULT_RISK_FREE_RATE
from options_trader.data.history import get_history
from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.composite import CompositeEventSource
from options_trader.data.events.fomc_source import FomcCalendarSource
from options_trader.data.events.yfinance_source import YFinanceEarningsSource
from options_trader.backtest.option_implied_beta_backtest import (
    HistoricalEodPdf,
    _spot_lookup_from_ts,
)
from options_trader.backtest.option_implied_backtest import (
    HistoricalTickerPdf,
    rolling_log_score_backtest_option_implied,
)
from options_trader.backtest.blend_backtest import rolling_log_score_backtest_blend_event
from options_trader.universe.sampler import sample_tickers


logger = logging.getLogger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _paired_report(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> str:
    d = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    win = float((d > 0).mean())
    verdict = "ACCEPT (more accurate)" if (d.mean() > 0 and p_val < 0.05) else "not significant"
    return (f"  {name_a} vs {name_b}: diff={d.mean():+.4f}  t={t_stat:.3f}  "
            f"p={p_val:.4g}  win={win:.0%}  -> {verdict}")


def _build_index_pdf(start, end):
    print("Fetching index histories (SPY, IWM) ...", flush=True)
    spy = get_history("SPY", start, end)
    iwm = get_history("IWM", start, end)
    return spy, iwm, HistoricalEodPdf(_spot_lookup_from_ts(spy, iwm), risk_free_rate=DEFAULT_RISK_FREE_RATE)


def _event_source():
    return CompositeEventSource(YFinanceEarningsSource(), FomcCalendarSource())


def _calendar(event_source, tkr, start, end):
    return EventCalendar.from_source(event_source, tkr, start, end + timedelta(days=60))


# ----------------------------------------------------------------------------------
# Mode 1: vs-eventboot  (items 1 + 3a)
# ----------------------------------------------------------------------------------

def run_vs_eventboot(args, tickers):
    spy, iwm, index_pdf = _build_index_pdf(args.start, args.end)
    event_source = _event_source()
    rows, n_fail = [], 0
    csv_path = f"{args.out_prefix}_eventboot.csv"

    for i, tkr in enumerate(tickers, 1):
        try:
            ts = get_history(tkr, args.start, args.end)
            cal = _calendar(event_source, tkr, args.start, args.end)
            out = rolling_log_score_backtest_blend_event(
                ts, spy, iwm, index_pdf, cal, horizon_days=args.horizon,
                holdout_days=args.holdout, n_paths=args.n_paths, base_seed=args.seed)
            eb, oi, bl = out["eboot"], out["oib"], out["blend"]
            for j in range(eb.n):
                e = eb.evaluations[j]
                rows.append({
                    "ticker": tkr, "forecast_date": e.forecast_date,
                    "realized_return": float(np.log(e.realized / e.spot)),
                    "ls_eboot": e.log_score,
                    "ls_oib": oi.evaluations[j].log_score,
                    "ls_blend": bl.evaluations[j].log_score,
                })
            adv = np.array([r["ls_oib"] - r["ls_eboot"] for r in rows if r["ticker"] == tkr])
            print(f"  [{i}/{len(tickers)}] {tkr}: n={eb.n} skip={bl.config['n_skipped']} "
                  f"eboot={eb.mean_log_score:.3f} oib={oi.mean_log_score:.3f} "
                  f"blend={bl.mean_log_score:.3f} (oib-eboot={adv.mean():+.3f})"
                  if eb.n else f"  [{i}/{len(tickers)}] {tkr}: no paired evals "
                               f"(skip={bl.config['n_skipped']})", flush=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [{i}/{len(tickers)}] {tkr}: FAILED ({type(exc).__name__}: {exc})", flush=True)

    if not rows:
        print("\nNo paired evaluations — cannot compare.")
        return
    df = pd.DataFrame(rows)
    eboot = df["ls_eboot"].to_numpy()
    oib = df["ls_oib"].to_numpy()
    blend = df["ls_blend"].to_numpy()

    # Per-ticker means (which tickers OIB / blend better on).
    g = df.groupby("ticker")
    per_tkr = pd.DataFrame({
        "n": g.size(),
        "eboot": g["ls_eboot"].mean(),
        "oib": g["ls_oib"].mean(),
        "blend": g["ls_blend"].mean(),
    })
    per_tkr["oib_minus_eboot"] = per_tkr["oib"] - per_tkr["eboot"]
    per_tkr["blend_minus_eboot"] = per_tkr["blend"] - per_tkr["eboot"]
    per_tkr = per_tkr.sort_values("oib_minus_eboot")

    _plot_compare(df, per_tkr, f"{args.out_prefix}_compare.png", args)

    oib_wins = int((per_tkr["oib"] > per_tkr["eboot"]).sum())
    blend_wins = int((per_tkr["blend"] > per_tkr["eboot"]).sum())
    lines = [
        "=" * 74,
        f"POOLED  n={len(df)} evals  tickers={df['ticker'].nunique()} (failed={n_fail})  "
        f"h={args.horizon} holdout={args.holdout} n_paths={args.n_paths}",
        f"  mean log score  event_bootstrap : {eboot.mean():.4f}",
        f"  mean log score  oib (event)     : {oib.mean():.4f}",
        f"  mean log score  blend 50/50     : {blend.mean():.4f}",
        "",
        "ITEM 3a — is the blend more accurate?",
        _paired_report("blend ", blend, "event_bootstrap", eboot),
        _paired_report("blend ", blend, "oib (event)    ", oib),
        _paired_report("oib   ", oib, "event_bootstrap", eboot),
        "",
        f"  per-ticker oib   beats event_bootstrap : {oib_wins}/{len(per_tkr)}",
        f"  per-ticker blend beats event_bootstrap : {blend_wins}/{len(per_tkr)}",
        "=" * 74,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    with open(f"{args.out_prefix}_blend_vs_eventboot.md", "w") as fh:
        fh.write("# 50/50 blend (event-bootstrap ⊕ event-OIB): comparison + accuracy\n\n")
        fh.write(f"Window {args.start}..{args.end}, h={args.horizon}, holdout={args.holdout}, "
                 f"n_paths={args.n_paths}, sample={args.sample} seed={args.sample_seed}\n\n")
        fh.write("## Per ticker (mean log score; sorted by oib−eboot)\n\n")
        for tkr, r in per_tkr.iterrows():
            fh.write(f"- {tkr}: n={int(r['n'])} eboot={r['eboot']:.3f} oib={r['oib']:.3f} "
                     f"blend={r['blend']:.3f} (oib−eboot={r['oib_minus_eboot']:+.3f}, "
                     f"blend−eboot={r['blend_minus_eboot']:+.3f})\n")
        fh.write("\n## Pooled\n\n```\n" + report + "\n```\n")
    print(f"\nWrote {args.out_prefix}_compare.png, _blend_vs_eventboot.md, and {csv_path}")


def _plot_compare(df, per_tkr, out_png, args):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    eboot, oib, blend = df["ls_eboot"], df["ls_oib"], df["ls_blend"]

    # (0,0) overlaid per-eval log-score distributions: event-bootstrap vs OIB.
    lo, hi = np.percentile(np.concatenate([eboot, oib]), [1, 99])
    bins = np.linspace(lo, hi, 50)
    axes[0, 0].hist(eboot, bins=bins, alpha=0.55, color="#1f77b4", label=f"event_bootstrap (mean {eboot.mean():.2f})")
    axes[0, 0].hist(oib, bins=bins, alpha=0.55, color="#ff7f0e", label=f"oib event (mean {oib.mean():.2f})")
    axes[0, 0].set_title("Per-eval log-score distribution: event-bootstrap vs OIB")
    axes[0, 0].set_xlabel("log score (higher = more accurate)"); axes[0, 0].set_ylabel("evals")
    axes[0, 0].legend()

    # (0,1) per-eval advantage (oib − eboot): where one wins.
    d = (oib - eboot).to_numpy()
    dlo, dhi = np.percentile(d, [1, 99])
    axes[0, 1].hist(np.clip(d, dlo, dhi), bins=50, color="#2ca02c", alpha=0.8)
    axes[0, 1].axvline(0, color="k", lw=0.8, ls=":")
    axes[0, 1].axvline(d.mean(), color="r", lw=1.2, label=f"mean {d.mean():+.3f}")
    axes[0, 1].axvline(np.median(d), color="b", lw=1.2, ls="--", label=f"median {np.median(d):+.3f}")
    axes[0, 1].set_title(f"oib − event_bootstrap per eval (OIB wins {(d>0).mean():.0%})")
    axes[0, 1].set_xlabel("log-score advantage (oib − eboot)"); axes[0, 1].legend()

    # (1,0) per-ticker mean (oib − eboot): which tickers OIB is better on.
    y = np.arange(len(per_tkr))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in per_tkr["oib_minus_eboot"]]
    axes[1, 0].barh(y, per_tkr["oib_minus_eboot"], color=colors, alpha=0.85)
    axes[1, 0].set_yticks(y); axes[1, 0].set_yticklabels(per_tkr.index, fontsize=7)
    axes[1, 0].axvline(0, color="k", lw=0.8)
    axes[1, 0].set_title("Per-ticker mean (oib − event_bootstrap)\ngreen = OIB more accurate")
    axes[1, 0].set_xlabel("mean log-score advantage")

    # (1,1) the blend's advantage over event-bootstrap (item 3a, visual).
    db = (blend - eboot).to_numpy()
    blo, bhi = np.percentile(db, [1, 99])
    axes[1, 1].hist(np.clip(db, blo, bhi), bins=50, color="#9467bd", alpha=0.8)
    axes[1, 1].axvline(0, color="k", lw=0.8, ls=":")
    axes[1, 1].axvline(db.mean(), color="r", lw=1.2, label=f"mean {db.mean():+.3f}")
    axes[1, 1].set_title(f"blend − event_bootstrap per eval (blend wins {(db>0).mean():.0%})")
    axes[1, 1].set_xlabel("log-score advantage (blend − eboot)"); axes[1, 1].legend()

    fig.suptitle(f"event-bootstrap vs event-OIB vs 50/50 blend — h={args.horizon}, "
                 f"holdout={args.holdout}, n={len(df)} evals / {df['ticker'].nunique()} tickers",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


# ----------------------------------------------------------------------------------
# Mode 2: vs-options  (item 3b)
# ----------------------------------------------------------------------------------

def run_vs_options(args, tickers):
    spy, iwm, index_pdf = _build_index_pdf(args.start, args.end)
    event_source = _event_source()
    rows, n_fail = [], 0
    csv_path = f"{args.out_prefix}_options.csv"

    for i, tkr in enumerate(tickers, 1):
        try:
            ts = get_history(tkr, args.start, args.end)
            cal = _calendar(event_source, tkr, args.start, args.end)
            blend_out = rolling_log_score_backtest_blend_event(
                ts, spy, iwm, index_pdf, cal, horizon_days=args.horizon,
                holdout_days=args.holdout, n_paths=args.n_paths, base_seed=args.seed)
            ticker_pdf = HistoricalTickerPdf(_spot_lookup_from_ts(ts), risk_free_rate=DEFAULT_RISK_FREE_RATE)
            opt = rolling_log_score_backtest_option_implied(
                ts, ticker_pdf, horizon_days=args.horizon, holdout_days=args.holdout,
                n_paths=args.n_paths, base_seed=args.seed)

            opt_by = {e.forecast_date: e.log_score for e in opt.evaluations}
            bl = blend_out["blend"]
            paired = [(e.log_score, opt_by[e.forecast_date]) for e in bl.evaluations
                      if e.forecast_date in opt_by]
            if not paired:
                print(f"  [{i}/{len(tickers)}] {tkr}: no common evals "
                      f"(blend_skip={bl.config['n_skipped']}, opt_skip={opt.config['n_skipped']})",
                      flush=True)
                continue
            for ls_blend, ls_opt in paired:
                rows.append({"ticker": tkr, "ls_blend": ls_blend, "ls_option_implied": ls_opt})
            a = np.array([p[0] for p in paired]); b = np.array([p[1] for p in paired])
            print(f"  [{i}/{len(tickers)}] {tkr}: n={len(paired)} "
                  f"blend={a.mean():.3f} option_implied={b.mean():.3f} "
                  f"(blend-opt={a.mean()-b.mean():+.3f})", flush=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [{i}/{len(tickers)}] {tkr}: FAILED ({type(exc).__name__}: {exc})", flush=True)

    if not rows:
        print("\nNo common evaluations — cannot compare.")
        return
    df = pd.DataFrame(rows)
    blend = df["ls_blend"].to_numpy(); opt = df["ls_option_implied"].to_numpy()
    per = df.groupby("ticker").apply(lambda x: x["ls_blend"].mean() - x["ls_option_implied"].mean(),
                                     include_groups=False)
    blend_wins = int((per > 0).sum())

    # Histogram of (blend − option_implied).
    fig, ax = plt.subplots(figsize=(9, 6))
    d = blend - opt
    dlo, dhi = np.percentile(d, [1, 99])
    ax.hist(np.clip(d, dlo, dhi), bins=50, color="#9467bd", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":"); ax.axvline(d.mean(), color="r", lw=1.2,
                                                         label=f"mean {d.mean():+.3f}")
    ax.set_title(f"blend − option_implied per eval (blend wins {(d>0).mean():.0%}, n={len(df)})")
    ax.set_xlabel("log-score advantage (blend − option_implied / market price)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_vs_options.png", dpi=130)
    print(f"Wrote {args.out_prefix}_vs_options.png")

    lines = [
        "=" * 74,
        f"POOLED  n={len(df)} evals  tickers={df['ticker'].nunique()} (failed={n_fail})  "
        f"h={args.horizon} holdout={args.holdout}",
        f"  mean log score  blend 50/50            : {blend.mean():.4f}",
        f"  mean log score  option_implied (market): {opt.mean():.4f}",
        "",
        "ITEM 3b — is the blend more accurate than the ticker's own option price?",
        _paired_report("blend", blend, "option_implied", opt),
        f"  per-ticker blend beats option_implied : {blend_wins}/{df['ticker'].nunique()}",
        "=" * 74,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(f"{args.out_prefix}_blend_vs_options.md", "w") as fh:
        fh.write("# 50/50 blend vs the ticker's own option price (option-implied)\n\n")
        fh.write(f"Window {args.start}..{args.end}, h={args.horizon}, holdout={args.holdout}, "
                 f"n_paths={args.n_paths}, sample={args.sample} seed={args.sample_seed}\n\n")
        fh.write("```\n" + report + "\n```\n")
    print(f"Wrote {args.out_prefix}_blend_vs_options.md and {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["vs-eventboot", "vs-options"], default="vs-eventboot")
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--sample-seed", type=int, default=11)
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--start", type=_parse_date, default=date(2022, 6, 1))
    ap.add_argument("--end", type=_parse_date, default=date(2025, 6, 1))
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--holdout", type=int, default=252)
    ap.add_argument("--n-paths", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="output/blend")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    tickers = args.tickers or sample_tickers(
        args.sample, seed=args.sample_seed, power=args.power, exclude=("SPY", "IWM"))
    print(f"Mode {args.mode} | Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Window [{args.start} .. {args.end}]  h={args.horizon}  holdout={args.holdout}  "
          f"n_paths={args.n_paths}\n")

    if args.mode == "vs-eventboot":
        run_vs_eventboot(args, tickers)
    else:
        run_vs_options(args, tickers)


if __name__ == "__main__":
    main()
