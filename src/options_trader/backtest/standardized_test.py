"""
standardized_test.py — the standardized multi-ticker forecaster benchmark.

WHAT IT IS
    A fixed, reproducible procedure for comparing forecasters across a test set
    of tickers using the rolling out-of-sample log score (see log_score.py). The
    test set is META (the V1 production ticker) plus N tickers drawn cap-weighted
    from a frozen S&P 500 snapshot (see universe/). Same snapshot + same sampler
    seed => same test set => same numbers, every run.

WHAT IT MEASURES
    For each ticker and each forecaster it runs rolling_log_score_backtest and
    keeps the per-day ForecastEvaluations. The headline question is the
    serial-dependence one: does BlockBootstrap (contiguous K-day blocks) beat
    plain Bootstrap (iid daily draws)? Gaussian is carried as the BS baseline.

SEEDING (two deliberate choices)
    * Per-iteration seed bump: within one rolling backtest the forecaster is
      rebuilt each day with seed = base_seed + iteration. A single fixed seed
      across iterations (as the original log_score harness used) produces
      autocorrelated Monte-Carlo noise that understates the standard error of the
      mean log score. Bumping decorrelates it.
    * Common random numbers ACROSS forecasters: every forecaster in a given run
      shares the same base_seed, so on any given day BlockBootstrap and Bootstrap
      see the same RNG stream. This is common-random-number variance reduction:
      the MC noise is correlated between the two, so the *paired* difference has
      much lower variance than if they were independently seeded.

AGGREGATION
    Per ticker: mean log score per forecaster.
    Pooled paired comparison (a vs b): within each ticker the two forecasters'
    evaluations align index-for-index (identical rolling dates), so we take the
    per-forecast log-score difference, pool across tickers, and run a one-sample
    t-test of mean difference vs 0. Win rate = fraction of tickers where a's mean
    log score exceeds b's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from options_trader.backtest.anchor import (
    ANCHOR_FNS,
    MA_BLEND_ANCHOR_FNS,
    AnchorFn,
)
from options_trader.backtest.log_score import (
    BacktestResult,
    rolling_log_score_backtest,
    rolling_log_score_backtest_conditioned,
    rolling_log_score_backtest_garch_fhs_event,
)
from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.source import EventSource, EventSourceError
from options_trader.data.history import HistoryError, get_history
from options_trader.forecast.base import Forecaster
from options_trader.forecast.block_bootstrap_forecaster import BlockBootstrapForecaster
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.drift_dampened_bootstrap_forecaster import (
    DriftDampenedBootstrapForecaster,
)
from options_trader.forecast.garch_fhs_forecaster import GarchFhsForecaster
from options_trader.forecast.gaussian_forecaster import GaussianForecaster
from options_trader.forecast.vol_gated_forecaster import VolGatedForecaster
from options_trader.forecast.return_distribution import (
    DEFAULT_DECAY_LAMBDA,
    ReturnDistribution,
)


logger = logging.getLogger(__name__)


# Registry of forecasters the standardized test knows how to build. Each class
# shares the constructor signature (distribution, n_paths, seed).
FORECASTER_CLASSES: dict[str, type[Forecaster]] = {
    "block_bootstrap": BlockBootstrapForecaster,
    "bootstrap": BootstrapForecaster,
    "gaussian": GaussianForecaster,
    "garch_fhs": GarchFhsForecaster,
}
DEFAULT_FORECASTERS = ("block_bootstrap", "bootstrap", "gaussian")
DEFAULT_BASE_SEED = 42

# Event-conditioned forecasters: handled via separate code paths (they need an
# EventCalendar + per-iteration conditioned distributions, not the uniform
# (distribution, n_paths, seed) factory), so they are NOT in FORECASTER_CLASSES.
EVENT_FORECASTER_NAME = "event_bootstrap"
GARCH_FHS_EVENT_FORECASTER_NAME = "garch_fhs_events"


def _decorrelated_factory(
    cls: type[Forecaster], n_paths: int, base_seed: int
) -> Callable[[ReturnDistribution], Forecaster]:
    """Factory that rebuilds `cls` each call with an incrementing seed.

    seed = base_seed + call_index. Decorrelates MC noise across rolling
    iterations; sharing base_seed across forecasters gives common random numbers.
    """
    counter = {"i": 0}

    def factory(rd: ReturnDistribution) -> Forecaster:
        seed = base_seed + counter["i"]
        counter["i"] += 1
        return cls(rd, n_paths=n_paths, seed=seed)

    return factory


@dataclass
class StandardizedTestResult:
    """Collected backtests across the test set, with aggregation helpers."""

    config: dict
    results: dict[str, dict[str, BacktestResult]]   # ticker -> forecaster -> result
    failures: dict[str, str] = field(default_factory=dict)  # ticker[:forecaster] -> error

    @property
    def tickers(self) -> list[str]:
        return list(self.results.keys())

    @property
    def forecasters(self) -> list[str]:
        for per in self.results.values():
            return list(per.keys())
        return []

    def summary_table(self) -> pd.DataFrame:
        """Per-ticker mean log score for each forecaster (rows=tickers)."""
        rows = []
        for ticker, per in self.results.items():
            row = {"ticker": ticker}
            for name, br in per.items():
                row[name] = br.mean_log_score
            row["n"] = next(iter(per.values())).n if per else 0
            rows.append(row)
        return pd.DataFrame(rows).set_index("ticker")

    def overall(self) -> pd.DataFrame:
        """Pooled mean log score per forecaster across ALL forecasts (every
        ticker-day), with std-error of the pooled mean."""
        rows = []
        for name in self.forecasters:
            scores = np.concatenate(
                [self.results[t][name].log_scores for t in self.results if name in self.results[t]]
            )
            rows.append({
                "forecaster": name,
                "n_forecasts": len(scores),
                "mean_log_score": float(scores.mean()),
                "stderr": float(scores.std(ddof=1) / np.sqrt(len(scores))),
            })
        return pd.DataFrame(rows).set_index("forecaster")

    def paired_comparison(self, a: str, b: str) -> dict:
        """Pooled paired comparison of forecaster `a` vs `b` (higher log score is
        better). Pairs by ticker-day, pools across tickers."""
        diffs: list[np.ndarray] = []
        per_ticker_win = []
        for ticker, per in self.results.items():
            if a not in per or b not in per:
                continue
            sa, sb = per[a].log_scores, per[b].log_scores
            m = min(len(sa), len(sb))   # identical in practice; guard anyway
            diffs.append(sa[:m] - sb[:m])
            per_ticker_win.append(per[a].mean_log_score > per[b].mean_log_score)
        if not diffs:
            raise ValueError(f"no tickers have both {a!r} and {b!r}")
        pooled = np.concatenate(diffs)
        t_res = stats.ttest_1samp(pooled, 0.0)
        return {
            "a": a,
            "b": b,
            "n_pairs": int(len(pooled)),
            "mean_diff": float(pooled.mean()),
            "t_stat": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
            "win_rate_by_ticker": float(np.mean(per_ticker_win)),
            "n_tickers": int(len(per_ticker_win)),
        }

    def to_report(self, headline: tuple[str, str] = ("block_bootstrap", "bootstrap")) -> str:
        """Render a human-readable markdown report."""
        lines = ["# Standardized Forecaster Test\n"]
        cfg = self.config
        lines.append(
            f"- Test set: {len(self.tickers)} tickers "
            f"(fixed: {cfg.get('fixed_tickers')}, sampled: {cfg.get('n_sampled')} "
            f"@ seed={cfg.get('sampler_seed')}, power={cfg.get('power')})"
        )
        lines.append(f"- Snapshot: {cfg.get('snapshot_path')}")
        lines.append(
            f"- horizon={cfg.get('horizon_days')}d, holdout={cfg.get('holdout_days')}d, "
            f"n_paths={cfg.get('n_paths')}, λ={cfg.get('decay_lambda')}, base_seed={cfg.get('base_seed')}\n"
        )
        lines.append("## Pooled mean log score (higher is better)\n")
        lines.append(self.overall().round(4).to_markdown())
        a, b = headline
        if a in self.forecasters and b in self.forecasters:
            pc = self.paired_comparison(a, b)
            lines.append(f"\n## Headline: {a} vs {b}\n")
            lines.append(
                f"- mean log-score diff = {pc['mean_diff']:+.4f} "
                f"(t={pc['t_stat']:.2f}, p={pc['p_value']:.3g}, n={pc['n_pairs']} paired forecasts)"
            )
            verdict = "indistinguishable" if pc["p_value"] > 0.05 else (
                f"{a} better" if pc["mean_diff"] > 0 else f"{b} better")
            lines.append(f"- per-ticker win rate ({a} > {b}): "
                         f"{pc['win_rate_by_ticker']:.0%} of {pc['n_tickers']} tickers")
            lines.append(f"- **verdict: {verdict}** at α=0.05\n")
        lines.append("## Per-ticker mean log score\n")
        lines.append(self.summary_table().round(4).to_markdown())
        if self.failures:
            lines.append("\n## Failures\n")
            for k, v in self.failures.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def run_standardized_test(
    tickers: list[str],
    forecasters: tuple[str, ...] = DEFAULT_FORECASTERS,
    horizon_days: int = 5,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = DEFAULT_BASE_SEED,
    start: Optional[date] = None,
    end: Optional[date] = None,
    event_source: Optional[EventSource] = None,
    event_window: int = 0,
    config_extra: Optional[dict] = None,
) -> StandardizedTestResult:
    """Run the standardized backtest for every (ticker, forecaster) pair.

    Tickers whose history can't be loaded, or that have too few bars for the
    requested holdout/horizon, are recorded in `.failures` and skipped — the test
    set is the drawn set minus failures (we never silently resample to backfill,
    which would break reproducibility).

    The `event_bootstrap` forecaster requires `event_source` to be supplied; for
    each ticker it builds an EventCalendar and runs the event-conditioned backtest.
    """
    _event_names = {EVENT_FORECASTER_NAME, GARCH_FHS_EVENT_FORECASTER_NAME}
    for name in forecasters:
        if name in _event_names:
            if event_source is None:
                raise ValueError(f"{name!r} requires an event_source")
        elif name not in FORECASTER_CLASSES:
            raise ValueError(
                f"unknown forecaster {name!r}; known: "
                f"{list(FORECASTER_CLASSES) + list(_event_names)}"
            )

    results: dict[str, dict[str, BacktestResult]] = {}
    failures: dict[str, str] = {}

    for ticker in tickers:
        try:
            ts = get_history(ticker, start=start, end=end)
        except HistoryError as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipping %s: %s", ticker, exc)
            continue

        per: dict[str, BacktestResult] = {}
        for name in forecasters:
            try:
                if name == EVENT_FORECASTER_NAME:
                    per[name] = _run_event_backtest(
                        ts, event_source, horizon_days, holdout_days,
                        decay_lambda, n_paths, base_seed, event_window,
                    )
                elif name == GARCH_FHS_EVENT_FORECASTER_NAME:
                    per[name] = _run_garch_fhs_event_backtest(
                        ts, event_source, horizon_days, holdout_days,
                        decay_lambda, n_paths, base_seed, event_window,
                    )
                else:
                    factory = _decorrelated_factory(FORECASTER_CLASSES[name], n_paths, base_seed)
                    per[name] = rolling_log_score_backtest(
                        ts, factory,
                        horizon_days=horizon_days,
                        holdout_days=holdout_days,
                        decay_lambda=decay_lambda,
                        name=f"{ticker}:{name}",
                    )
            except (ValueError, EventSourceError, np.linalg.LinAlgError) as exc:
                failures[f"{ticker}:{name}"] = f"{type(exc).__name__}: {exc}"
                logger.warning("backtest failed %s:%s: %s", ticker, name, exc)
        if per:
            results[ticker] = per

    config = {
        "tickers": tickers,
        "forecasters": list(forecasters),
        "horizon_days": horizon_days,
        "holdout_days": holdout_days,
        "decay_lambda": decay_lambda,
        "n_paths": n_paths,
        "base_seed": base_seed,
        "event_window": event_window,
    }
    if config_extra:
        config.update(config_extra)
    logger.info("standardized test complete: %d/%d tickers succeeded",
                len(results), len(tickers))
    return StandardizedTestResult(config=config, results=results, failures=failures)


DEFAULT_ANCHOR_HEADLINE = ("intraday_random", "stale_close")
# Hypothesis 1: 50/50 fresh-price + 7-day MA vs the live incumbent (fresh price).
MA_BLEND_ANCHOR_HEADLINE = ("ma7_blend", "intraday_random")


def run_anchor_experiment(
    tickers: list[str],
    horizon_days: int = 5,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = DEFAULT_BASE_SEED,
    anchor_seed: int = 12345,
    start: Optional[date] = None,
    end: Optional[date] = None,
    anchor_fns: Optional[dict[str, AnchorFn]] = None,
    config_extra: Optional[dict] = None,
) -> StandardizedTestResult:
    """Measure whether the choice of spot anchor changes forecast log score.

    Holds the forecaster fixed (BootstrapForecaster) and varies ONLY the spot
    anchor, running the rolling log-score backtest once per anchor in
    ``anchor_fns`` (default ``ANCHOR_FNS``: 'stale_close' incumbent vs
    'intraday_random' candidate; pass ``MA_BLEND_ANCHOR_FNS`` for the
    moving-average-blend experiment). The anchors are packed as the "forecaster"
    keys of a StandardizedTestResult, so the existing pooled paired t-test
    (paired_comparison / to_report) applies unchanged.

    Common random numbers: both anchors share ``base_seed`` for the forecaster's
    Monte-Carlo draws (a fresh per-arm decorrelated factory both start their
    counter at 0), so on any ticker-day the two arms see the SAME return draws
    and the paired log-score difference isolates the anchor effect.

    Faithfulness: each iteration models running intraday on the day after a
    completed close — the stale anchor is grid-aligned but a day old, the
    intraday anchor is fresh but off-grid by a partial day. Both forecast the
    same horizon to the same realised close[t+horizon] (see backtest/anchor.py).
    """
    fns = anchor_fns or ANCHOR_FNS
    results: dict[str, dict[str, BacktestResult]] = {}
    failures: dict[str, str] = {}

    for ticker in tickers:
        try:
            ts = get_history(ticker, start=start, end=end)
        except HistoryError as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipping %s: %s", ticker, exc)
            continue

        per: dict[str, BacktestResult] = {}
        for anchor_name, anchor_fn in fns.items():
            try:
                # Fresh factory per arm => both counters start at 0 => CRN across arms.
                factory = _decorrelated_factory(BootstrapForecaster, n_paths, base_seed)
                per[anchor_name] = rolling_log_score_backtest(
                    ts, factory,
                    horizon_days=horizon_days,
                    holdout_days=holdout_days,
                    decay_lambda=decay_lambda,
                    anchor_fn=anchor_fn,
                    anchor_seed=anchor_seed,
                    name=f"{ticker}:{anchor_name}",
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                failures[f"{ticker}:{anchor_name}"] = f"{type(exc).__name__}: {exc}"
                logger.warning("anchor backtest failed %s:%s: %s", ticker, anchor_name, exc)
        if per:
            results[ticker] = per

    config = {
        "experiment": "anchor",
        "tickers": tickers,
        "forecasters": list(fns.keys()),
        "horizon_days": horizon_days,
        "holdout_days": holdout_days,
        "decay_lambda": decay_lambda,
        "n_paths": n_paths,
        "base_seed": base_seed,
        "anchor_seed": anchor_seed,
    }
    if config_extra:
        config.update(config_extra)
    logger.info("anchor experiment complete: %d/%d tickers succeeded",
                len(results), len(tickers))
    return StandardizedTestResult(config=config, results=results, failures=failures)


# Default κ grid for the drift-dampening sweep (0.0 == plain bootstrap baseline).
DEFAULT_DAMPENING_KAPPAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _dampening_arm_name(kappa: float) -> str:
    return f"dampen_{kappa:.2f}"


DAMPENING_BASELINE_ARM = _dampening_arm_name(0.0)


def _dampening_factory(
    n_paths: int, base_seed: int, kappa: float
) -> Callable[[ReturnDistribution], Forecaster]:
    """Decorrelated factory building a DriftDampenedBootstrapForecaster at a fixed κ.

    Seed scheme matches `_decorrelated_factory`: a fresh factory per arm starts its
    counter at 0, so every κ-arm shares the same RNG stream on a given ticker-day
    (common random numbers). The κ=0 arm is therefore byte-identical to the plain
    BootstrapForecaster baseline, and the paired log-score difference isolates the
    drift-dampening effect alone.
    """
    counter = {"i": 0}

    def factory(rd: ReturnDistribution) -> Forecaster:
        seed = base_seed + counter["i"]
        counter["i"] += 1
        return DriftDampenedBootstrapForecaster(
            rd, n_paths=n_paths, seed=seed, drift_dampening=kappa
        )

    return factory


def run_drift_dampening_experiment(
    tickers: list[str],
    kappas: tuple[float, ...] = DEFAULT_DAMPENING_KAPPAS,
    horizon_days: int = 5,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = DEFAULT_BASE_SEED,
    start: Optional[date] = None,
    end: Optional[date] = None,
    config_extra: Optional[dict] = None,
) -> StandardizedTestResult:
    """Sweep the drift-dampening fraction κ on the bootstrap forecaster.

    For each κ in `kappas` runs the rolling log-score backtest with a
    DriftDampenedBootstrapForecaster (terminal mean shrunk toward spot by κ via a
    pure linear translation). Each κ is packed as a "forecaster" arm
    ('dampen_0.00', 'dampen_0.50', …) of a StandardizedTestResult, so the existing
    pooled paired t-test (paired_comparison / to_report) applies unchanged. The
    κ=0 arm IS the plain bootstrap incumbent (zero shift), and shares common random
    numbers with every other arm, so `paired_comparison('dampen_0.50', 'dampen_0.00')`
    is the decision-criterion test for that κ vs the incumbent.

    Fit κ by reading `overall()` (pooled mean log score per arm) for the best κ and
    `paired_comparison(best, 'dampen_0.00')` for its significance vs the incumbent.
    """
    if 0.0 not in kappas:
        # The baseline must be present so every arm has a paired incumbent.
        kappas = (0.0, *kappas)

    results: dict[str, dict[str, BacktestResult]] = {}
    failures: dict[str, str] = {}

    for ticker in tickers:
        try:
            ts = get_history(ticker, start=start, end=end)
        except HistoryError as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipping %s: %s", ticker, exc)
            continue

        per: dict[str, BacktestResult] = {}
        for kappa in kappas:
            arm = _dampening_arm_name(kappa)
            try:
                factory = _dampening_factory(n_paths, base_seed, kappa)
                per[arm] = rolling_log_score_backtest(
                    ts, factory,
                    horizon_days=horizon_days,
                    holdout_days=holdout_days,
                    decay_lambda=decay_lambda,
                    name=f"{ticker}:{arm}",
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                failures[f"{ticker}:{arm}"] = f"{type(exc).__name__}: {exc}"
                logger.warning("dampening backtest failed %s:%s: %s", ticker, arm, exc)
        if per:
            results[ticker] = per

    config = {
        "experiment": "drift_dampening",
        "tickers": tickers,
        "forecasters": [_dampening_arm_name(k) for k in kappas],
        "kappas": list(kappas),
        "horizon_days": horizon_days,
        "holdout_days": holdout_days,
        "decay_lambda": decay_lambda,
        "n_paths": n_paths,
        "base_seed": base_seed,
    }
    if config_extra:
        config.update(config_extra)
    logger.info("drift-dampening experiment complete: %d/%d tickers succeeded",
                len(results), len(tickers))
    return StandardizedTestResult(config=config, results=results, failures=failures)


def best_dampening_arm(result: StandardizedTestResult) -> tuple[str, float]:
    """The κ-arm with the highest pooled mean log score, and its score.

    Used to pick the fitted κ and form the headline (best vs the κ=0 incumbent).
    """
    overall = result.overall()
    best = overall["mean_log_score"].idxmax()
    return best, float(overall.loc[best, "mean_log_score"])


# Default τ grid for the vol-gate sweep. 0.0 == always FHS; inf == always bootstrap
# (the incumbent). The interior values gate FHS to increasingly-divergent vol regimes.
DEFAULT_GATE_TAUS = (0.0, 0.9, 1.0, 1.1, 1.25, 1.5, float("inf"))


def _gate_arm_name(tau: float) -> str:
    return f"gate_{tau:.2f}" if tau != float("inf") else "gate_inf"


GATE_ALWAYS_FHS_ARM = _gate_arm_name(0.0)        # == GarchFhsForecaster
GATE_BOOTSTRAP_ARM = _gate_arm_name(float("inf"))  # == BootstrapForecaster (incumbent)


def _vol_gate_factory(
    n_paths: int, base_seed: int, tau: float
) -> Callable[[ReturnDistribution], Forecaster]:
    """Decorrelated factory building a VolGatedForecaster at a fixed threshold τ.
    Same seed scheme as `_decorrelated_factory` → CRN across τ-arms and with the
    pure bootstrap / FHS arms."""
    counter = {"i": 0}

    def factory(rd: ReturnDistribution) -> Forecaster:
        seed = base_seed + counter["i"]
        counter["i"] += 1
        return VolGatedForecaster(rd, n_paths=n_paths, seed=seed, vol_threshold=tau)

    return factory


def run_vol_gate_experiment(
    tickers: list[str],
    taus: tuple[float, ...] = DEFAULT_GATE_TAUS,
    horizon_days: int = 21,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = DEFAULT_BASE_SEED,
    start: Optional[date] = None,
    end: Optional[date] = None,
    config_extra: Optional[dict] = None,
) -> StandardizedTestResult:
    """Sweep the vol-gate threshold τ: route each ticker-day to GARCH-FHS when
    σ_{T+1}/σ̄ ≥ τ, else the plain bootstrap.

    Each τ is a "forecaster" arm ('gate_0.00' … 'gate_inf') so the pooled paired
    t-test applies unchanged. The endpoints are the two pure forecasters:
    `gate_0.00` == always-FHS, `gate_inf` == always-bootstrap (incumbent). Fit τ by
    reading `overall()` for the best arm; compare it to BOTH `gate_inf` (does gating
    beat the bootstrap?) and `gate_0.00` (does gating beat plain FHS?).
    """
    # Ensure both endpoints are present so every interior τ has its two references.
    taus = tuple(taus)
    if 0.0 not in taus:
        taus = (0.0, *taus)
    if float("inf") not in taus:
        taus = (*taus, float("inf"))

    results: dict[str, dict[str, BacktestResult]] = {}
    failures: dict[str, str] = {}

    for ticker in tickers:
        try:
            ts = get_history(ticker, start=start, end=end)
        except HistoryError as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipping %s: %s", ticker, exc)
            continue

        per: dict[str, BacktestResult] = {}
        for tau in taus:
            arm = _gate_arm_name(tau)
            try:
                factory = _vol_gate_factory(n_paths, base_seed, tau)
                per[arm] = rolling_log_score_backtest(
                    ts, factory,
                    horizon_days=horizon_days,
                    holdout_days=holdout_days,
                    decay_lambda=decay_lambda,
                    name=f"{ticker}:{arm}",
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                failures[f"{ticker}:{arm}"] = f"{type(exc).__name__}: {exc}"
                logger.warning("vol-gate backtest failed %s:%s: %s", ticker, arm, exc)
        if per:
            results[ticker] = per

    config = {
        "experiment": "vol_gate",
        "tickers": tickers,
        "forecasters": [_gate_arm_name(t) for t in taus],
        "taus": list(taus),
        "horizon_days": horizon_days,
        "holdout_days": holdout_days,
        "decay_lambda": decay_lambda,
        "n_paths": n_paths,
        "base_seed": base_seed,
    }
    if config_extra:
        config.update(config_extra)
    logger.info("vol-gate experiment complete: %d/%d tickers succeeded",
                len(results), len(tickers))
    return StandardizedTestResult(config=config, results=results, failures=failures)


def best_gate_arm(result: StandardizedTestResult) -> tuple[str, float]:
    """The τ-arm with the highest pooled mean log score, and its score."""
    overall = result.overall()
    best = overall["mean_log_score"].idxmax()
    return best, float(overall.loc[best, "mean_log_score"])


def _run_event_backtest(
    ts, event_source, horizon_days, holdout_days, decay_lambda,
    n_paths, base_seed, event_window,
) -> BacktestResult:
    """Build the ticker's EventCalendar over its data window and run the
    event-conditioned rolling backtest."""
    from datetime import timedelta
    data_start = pd.Timestamp(ts.dates[0]).date()
    data_end = pd.Timestamp(ts.dates[-1]).date() + timedelta(days=1)
    calendar = EventCalendar.from_source(event_source, ts.ticker, data_start, data_end)
    return rolling_log_score_backtest_conditioned(
        ts, calendar,
        horizon_days=horizon_days,
        holdout_days=holdout_days,
        decay_lambda=decay_lambda,
        n_paths=n_paths,
        base_seed=base_seed,
        event_window=event_window,
        name=f"{ts.ticker}:{EVENT_FORECASTER_NAME}",
    )


def _run_garch_fhs_event_backtest(
    ts, event_source, horizon_days, holdout_days, decay_lambda,
    n_paths, base_seed, event_window,
) -> BacktestResult:
    """Build the ticker's EventCalendar and run the GARCH-FHS-event backtest."""
    from datetime import timedelta
    data_start = pd.Timestamp(ts.dates[0]).date()
    data_end = pd.Timestamp(ts.dates[-1]).date() + timedelta(days=1)
    calendar = EventCalendar.from_source(event_source, ts.ticker, data_start, data_end)
    return rolling_log_score_backtest_garch_fhs_event(
        ts, calendar,
        horizon_days=horizon_days,
        holdout_days=holdout_days,
        decay_lambda=decay_lambda,
        n_paths=n_paths,
        base_seed=base_seed,
        event_window=event_window,
        name=f"{ts.ticker}:{GARCH_FHS_EVENT_FORECASTER_NAME}",
    )
