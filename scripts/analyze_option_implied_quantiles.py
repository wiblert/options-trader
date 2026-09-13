#!/usr/bin/env python3
"""
analyze_option_implied_quantiles.py — WHERE (in the realized return) do we beat the
option market's own price?

Runs the option-implied (own-options) BENCHMARK and the production default
(event-bootstrap) over a sample of tickers, retains EVERY paired evaluation (the driver
script only keeps pooled means), then asks: bucketed by the realized horizon return, in
which quantile does our production forecast out-score the market-implied PDF?

Per evaluation we record:
  * realized_return = log(realized_close / spot)            (the actual h-day move)
  * ls_boot  = event-bootstrap (production) KDE log score
  * ls_oi    = option-implied (market price) KDE log score
  * advantage = ls_boot - ls_oi   (>0  => OUR forecast beats the market price here)

Outputs (default prefix output/option_implied_quantiles):
  * <prefix>.png  — (1) histogram of realized returns split by win/lose, with per-bin
                    win-rate; (2) mean log-score advantage by realized-return decile.
  * <prefix>.csv  — one row per paired evaluation (for ad-hoc re-analysis).
  * <prefix>.md   — text summary incl. the best/worst return quantile.

Network-backed and slow (one historical option-chain fetch per ticker per eval date,
cached per (ticker, run_date)). Dates with no liquid chain are skipped (counted), and
dates before Alpaca's Feb-2024 option history never have a PDF, so the effective scored
window starts there regardless of --start.

Usage:
    python scripts/analyze_option_implied_quantiles.py --sample 30 \
        --start 2022-06-01 --end 2025-06-01 --horizon 21 --holdout 378
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _collect_rows(tkr, ts, ticker_pdf, event_source, *, horizon, holdout, n_paths, seed):
    """Run both arms for one ticker; return date-aligned per-eval rows."""
    oi = rolling_log_score_backtest_option_implied(
        ts, ticker_pdf, horizon_days=horizon, holdout_days=holdout,
        n_paths=n_paths, base_seed=seed)
    cal = EventCalendar.from_source(event_source, tkr, ts_start(ts), end_plus(ts))
    boot = rolling_log_score_backtest_conditioned(
        ts, cal, horizon_days=horizon, holdout_days=holdout, n_paths=n_paths, base_seed=seed)

    boot_by = {e.forecast_date: e for e in boot.evaluations}
    rows = []
    for e in oi.evaluations:
        b = boot_by.get(e.forecast_date)
        if b is None:
            continue
        rows.append({
            "ticker": tkr,
            "forecast_date": e.forecast_date,
            "spot": e.spot,
            "realized": e.realized,
            "realized_return": float(np.log(e.realized / e.spot)),
            "pit_oi": e.percentile_of_realized,
            "pit_boot": b.percentile_of_realized,
            "ls_oi": e.log_score,
            "ls_boot": b.log_score,
            "advantage": b.log_score - e.log_score,
        })
    return rows, oi.config["n_skipped"]


def ts_start(ts):
    return pd.Timestamp(ts.dates[0]).date()


def end_plus(ts):
    return pd.Timestamp(ts.dates[-1]).date() + timedelta(days=60)


def _quantile_table(df, value="realized_return", n_bins=10):
    """Mean advantage + win rate per quantile bin of `value`. Returns a DataFrame."""
    # qcut on ranks to get equal-count bins even with ties.
    df = df.copy()
    df["bin"] = pd.qcut(df[value].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin")
    out = pd.DataFrame({
        "n": g.size(),
        "ret_lo": g[value].min(),
        "ret_hi": g[value].max(),
        "ret_mid": g[value].median(),
        "mean_advantage": g["advantage"].mean(),
        "median_advantage": g["advantage"].median(),
        "win_rate": g["advantage"].agg(lambda s: float((s > 0).mean())),
    }).reset_index()
    return out


def _plot(df, qt, out_png, *, horizon, holdout):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: histogram of realized returns, split by win (we beat market) / lose.
    win = df[df["advantage"] > 0]["realized_return"].to_numpy()
    lose = df[df["advantage"] <= 0]["realized_return"].to_numpy()
    lo, hi = np.percentile(df["realized_return"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 41)
    ax1.hist([win, lose], bins=bins, stacked=True, color=["#2ca02c", "#d62728"],
             label=[f"we beat market (n={len(win)})", f"market wins (n={len(lose)})"], alpha=0.85)
    ax1.axvline(0.0, color="k", lw=0.8, ls=":")
    ax1.set_xlabel(f"realized {horizon}-day log return")
    ax1.set_ylabel("evaluations")
    ax1.set_title("Where the return landed: our wins vs market wins")
    ax1.legend()

    # Panel 2: mean log-score advantage by realized-return decile.
    x = np.arange(len(qt))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in qt["mean_advantage"]]
    ax2.bar(x, qt["mean_advantage"], color=colors, alpha=0.85)
    ax2.axhline(0.0, color="k", lw=0.8)
    best = int(qt["mean_advantage"].idxmax())
    ax2.bar([best], [qt["mean_advantage"].iloc[best]], color="#1f77b4",
            edgecolor="k", lw=1.5, label="best decile")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{lo:+.0%}\n{hi:+.0%}" for lo, hi in zip(qt["ret_lo"], qt["ret_hi"])],
                        fontsize=8)
    ax2.set_xlabel("realized-return decile (range)")
    ax2.set_ylabel("mean log-score advantage  (boot − option_implied)")
    ax2.set_title("Our edge over the market price, by return quantile\n(>0 = we are more accurate)")
    ax2.legend()

    fig.suptitle(
        f"Option-implied (market price) vs production forecast — h={horizon}, holdout={holdout}, "
        f"n={len(df)} evals across {df['ticker'].nunique()} tickers", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--start", type=_parse_date, default=date(2022, 6, 1))
    ap.add_argument("--end", type=_parse_date, default=date(2025, 6, 1))
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--holdout", type=int, default=378)
    ap.add_argument("--n-paths", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out-prefix", default="output/option_implied_quantiles")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    tickers = args.tickers or sample_tickers(
        args.sample, seed=args.sample_seed, power=args.power, exclude=("SPY", "IWM"))
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Window [{args.start} .. {args.end}]  h={args.horizon}  holdout={args.holdout}  "
          f"n_paths={args.n_paths}")
    print("advantage = ls_boot - ls_oi  (>0 => our production forecast beats the market price)\n")

    event_source = CompositeEventSource(YFinanceEarningsSource(), FomcCalendarSource())
    all_rows = []
    n_fail = 0
    csv_path = f"{args.out_prefix}.csv"

    for i, tkr in enumerate(tickers, 1):
        try:
            ts = get_history(tkr, args.start, args.end)
            ticker_pdf = HistoricalTickerPdf(_spot_lookup_from_ts(ts), risk_free_rate=0.04)
            rows, skipped = _collect_rows(
                tkr, ts, ticker_pdf, event_source,
                horizon=args.horizon, holdout=args.holdout,
                n_paths=args.n_paths, seed=args.seed)
            all_rows.extend(rows)
            adv = np.array([r["advantage"] for r in rows]) if rows else np.array([])
            print(f"  [{i}/{len(tickers)}] {tkr}: paired={len(rows)} skip={skipped} "
                  f"mean_adv={adv.mean():+.3f} win={ (adv>0).mean():.0%}" if len(rows)
                  else f"  [{i}/{len(tickers)}] {tkr}: no paired evals (skip={skipped})", flush=True)
            # Persist incrementally so a long run is never lost.
            pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [{i}/{len(tickers)}] {tkr}: FAILED ({type(exc).__name__}: {exc})", flush=True)

    if not all_rows:
        print("\nNo paired evaluations across any ticker — cannot analyze.")
        return

    df = pd.DataFrame(all_rows)
    qt = _quantile_table(df, "realized_return", n_bins=args.bins)
    best = int(qt["mean_advantage"].idxmax())
    worst = int(qt["mean_advantage"].idxmin())

    _plot(df, qt, f"{args.out_prefix}.png", horizon=args.horizon, holdout=args.holdout)

    overall_adv = df["advantage"].mean()
    overall_win = float((df["advantage"] > 0).mean())
    lines = [
        "=" * 72,
        f"OVERALL  n={len(df)} evals  tickers={df['ticker'].nunique()} (failed={n_fail})  "
        f"h={args.horizon} holdout={args.holdout}",
        f"  mean advantage (boot - option_implied): {overall_adv:+.4f}   win rate: {overall_win:.1%}",
        f"  (>0 => our production forecast is more accurate than the market's own price)",
        "",
        "Log-score advantage by realized-return decile (1=most negative return):",
        f"  {'decile':>6} {'return range':>20} {'n':>6} {'mean_adv':>10} {'med_adv':>9} {'win%':>7}",
    ]
    for _, r in qt.iterrows():
        lines.append(f"  {int(r['bin'])+1:>6} {r['ret_lo']:>+9.1%}..{r['ret_hi']:<+8.1%} "
                     f"{int(r['n']):>6} {r['mean_advantage']:>+10.3f} {r['median_advantage']:>+9.3f} "
                     f"{r['win_rate']:>6.0%}")
    lines += [
        "",
        f"BEST  decile {best+1}: realized return {qt['ret_lo'].iloc[best]:+.1%}..{qt['ret_hi'].iloc[best]:+.1%} "
        f"-> mean advantage {qt['mean_advantage'].iloc[best]:+.3f}, win {qt['win_rate'].iloc[best]:.0%}",
        f"WORST decile {worst+1}: realized return {qt['ret_lo'].iloc[worst]:+.1%}..{qt['ret_hi'].iloc[worst]:+.1%} "
        f"-> mean advantage {qt['mean_advantage'].iloc[worst]:+.3f}, win {qt['win_rate'].iloc[worst]:.0%}",
        "=" * 72,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(f"{args.out_prefix}.md", "w") as fh:
        fh.write("# Option-implied vs production: edge by realized-return quantile\n\n")
        fh.write(f"Window {args.start}..{args.end}, h={args.horizon}, holdout={args.holdout}, "
                 f"n_paths={args.n_paths}, sample={args.sample} seed={args.sample_seed}\n\n")
        fh.write("```\n" + report + "\n```\n")
    print(f"Wrote {args.out_prefix}.md and {csv_path}")


if __name__ == "__main__":
    main()
