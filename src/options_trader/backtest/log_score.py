"""
log_score.py — rolling out-of-sample evaluation of forecasters.

For each day in a holdout window:
    1. Build a Forecaster from the current (incrementally-updated) ReturnDistribution
    2. Forecast K days ahead from that day's close
    3. Score the realised close at horizon against the forecast PDF
        - Percentile of realised (probability integral transform — PIT)
        - Log score: log p_forecast(realised), KDE-estimated density

The PIT array drives calibration diagnostics (regression vs y=x); the mean log
score is a strictly proper score for forecast quality across calibration AND
sharpness simultaneously.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from options_trader.backtest.anchor import AnchorContext, AnchorFn
from options_trader.data.events.calendar import EventCalendar
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.base import Forecaster
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions
from options_trader.forecast.event_bootstrap_forecaster import (
    EventConditionedBootstrapForecaster,
)
from options_trader.forecast.price_distribution import PriceDistribution
from options_trader.forecast.return_distribution import (
    DEFAULT_DECAY_LAMBDA,
    ReturnDistribution,
)


logger = logging.getLogger(__name__)


# Floor for log score to keep -inf out of the mean when realised lands in a
# region the forecast assigned ~0 density to.
MIN_DENSITY = 1e-12


@dataclass
class ForecastEvaluation:
    """One forecast vs one realisation."""
    forecast_date: date
    realized_date: date
    spot: float
    realized: float
    forecast_mean: float
    forecast_std: float
    percentile_of_realized: float
    log_score: float


@dataclass
class BacktestResult:
    """Collected evaluations from a rolling backtest, with summary helpers."""
    name: str
    evaluations: list[ForecastEvaluation]
    config: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.evaluations)

    @property
    def percentiles(self) -> np.ndarray:
        return np.array([e.percentile_of_realized for e in self.evaluations])

    @property
    def log_scores(self) -> np.ndarray:
        return np.array([e.log_score for e in self.evaluations])

    @property
    def mean_log_score(self) -> float:
        return float(np.mean(self.log_scores))

    @property
    def log_score_stderr(self) -> float:
        """Std-error of the mean log score (Newey-West NOT applied — paths may overlap)."""
        return float(np.std(self.log_scores, ddof=1) / np.sqrt(self.n))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self.evaluations])

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.name!r}, n={self.n}, "
            f"mean_log_score={self.mean_log_score:.3f}±{self.log_score_stderr:.3f})"
        )


# ---------- helpers ----------


def _weighted_cdf_at(pd_: PriceDistribution, value: float) -> float:
    """Weighted CDF of pd_ evaluated at `value`."""
    order = np.argsort(pd_.prices)
    return float(np.interp(value, pd_.prices[order], np.cumsum(pd_.weights[order])))


def _log_density(pd_: PriceDistribution, value: float) -> float:
    """Log density of pd_ at `value` via weighted Gaussian KDE.

    Density is clipped at MIN_DENSITY to avoid log(0) when realised lands in
    a region the forecast assigned vanishing probability.
    """
    kde = gaussian_kde(pd_.prices, weights=pd_.weights, bw_method="scott")
    density = float(kde(value)[0])
    return float(np.log(max(density, MIN_DENSITY)))


# ---------- main entry ----------


def rolling_log_score_backtest(
    ts: StockReturnTS,
    forecaster_factory: Callable[[ReturnDistribution], Forecaster],
    horizon_days: int = 5,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    name: Optional[str] = None,
    anchor_fn: Optional[AnchorFn] = None,
    anchor_seed: int = 12345,
) -> BacktestResult:
    """Run a rolling out-of-sample backtest.

    Args:
        ts: StockReturnTS with at least `holdout_days + horizon_days + 2` bars.
        forecaster_factory: callable that takes a ReturnDistribution and returns
            a Forecaster. Letting the caller build the forecaster means we can
            test any Forecaster subclass through this function (Bootstrap, JD, etc.).
        horizon_days: forecast horizon in trading days.
        holdout_days: number of out-of-sample evaluation days.
        decay_lambda: exponential-decay parameter passed to ReturnDistribution;
            also used as the default in add_sample updates.
        name: human-readable label for the result (used by plots).
        anchor_fn: optional spot-anchor source (see backtest/anchor.py). When
            None (default) the forecast anchors on close[t] — the historical
            behaviour, unchanged. When supplied, we model running INTRADAY on the
            day after close[t]: the anchor is chosen by anchor_fn from an
            AnchorContext (prev_close=close[t] plus the next day's OHLC), while
            the realised target stays close[t+horizon]. This lets the backtest
            measure stale-close vs intraday-price anchoring. Requires
            horizon_days >= 2 so the candidate's next-day bar precedes the target.
        anchor_seed: seed for the (separate) RNG that anchor_fn may draw from, so
            intraday-price draws are reproducible and independent of the
            forecaster's Monte-Carlo stream.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if holdout_days < 1:
        raise ValueError(f"holdout_days must be >= 1, got {holdout_days}")
    if anchor_fn is not None and horizon_days < 2:
        raise ValueError(
            "anchor_fn requires horizon_days >= 2 (the candidate anchors on the "
            "day after close[t], which must precede the realised target)"
        )

    close = np.asarray(ts.close, dtype=float)
    open_ = np.asarray(ts.open, dtype=float)
    high = np.asarray(ts.high, dtype=float)
    low = np.asarray(ts.low, dtype=float)
    dates = ts.dates
    anchor_rng = np.random.default_rng(anchor_seed) if anchor_fn is not None else None
    n_bars = len(close)
    min_required = holdout_days + horizon_days + 2
    if n_bars < min_required:
        raise ValueError(
            f"need at least {min_required} bars, got {n_bars} for {ts.ticker}"
        )

    log_returns = np.diff(np.log(close))                 # length n_bars - 1
    T = n_bars - holdout_days - 1                        # last training index
    initial_returns = log_returns[:T]                    # T returns, newest = log(close[T]/close[T-1])

    rd = ReturnDistribution(initial_returns, decay_lambda=decay_lambda, label=ts.ticker)

    evaluations: list[ForecastEvaluation] = []
    for t in range(T, n_bars - horizon_days):
        realized = float(close[t + horizon_days])
        if anchor_fn is None:
            spot = float(close[t])
        else:
            # Running intraday on the day AFTER close[t]: anchor on that day's
            # bar (index t+1), but still forecast/score to close[t+horizon].
            ctx = AnchorContext(
                prev_close=float(close[t]),
                today_open=float(open_[t + 1]),
                today_high=float(high[t + 1]),
                today_low=float(low[t + 1]),
                today_close=float(close[t + 1]),
                rng=anchor_rng,
                # Completed closes through index t (newest = prev_close). All
                # <= t, so a moving average over this is strictly point-in-time.
                trailing_closes=close[: t + 1],
            )
            spot = float(anchor_fn(ctx))

        # Defend against a factory that mutates rd (e.g. smooth_samples()) —
        # that would corrupt every later iteration sharing this same object.
        forecaster = forecaster_factory(copy.deepcopy(rd))
        fcast = forecaster.forecast(horizon_days=horizon_days, spot=spot)

        pct = _weighted_cdf_at(fcast, realized)
        log_score = _log_density(fcast, realized)

        evaluations.append(ForecastEvaluation(
            forecast_date=pd.Timestamp(dates[t]).date(),
            realized_date=pd.Timestamp(dates[t + horizon_days]).date(),
            spot=spot,
            realized=realized,
            forecast_mean=fcast.mean(),
            forecast_std=fcast.std(),
            percentile_of_realized=pct,
            log_score=log_score,
        ))

        # Advance the distribution by one day (log_returns[t] is log(close[t+1]/close[t]))
        if t + 1 < n_bars:
            rd.add_sample(float(log_returns[t]))

    config = {
        "ticker": ts.ticker,
        "horizon_days": horizon_days,
        "holdout_days": holdout_days,
        "decay_lambda": decay_lambda,
        "n_evaluations": len(evaluations),
    }
    label = name or f"{ts.ticker} h={horizon_days} λ={decay_lambda}"
    logger.info("rolling_log_score_backtest %s: %d evaluations", label, len(evaluations))
    return BacktestResult(name=label, evaluations=evaluations, config=config)


def _truncate_ts(ts: StockReturnTS, t: int) -> StockReturnTS:
    """Return a copy of `ts` keeping bars [0, t] inclusive (so log-returns are [0, t))."""
    s = slice(0, t + 1)
    return dataclasses.replace(
        ts, dates=ts.dates[s], open=ts.open[s], high=ts.high[s],
        low=ts.low[s], close=ts.close[s], volume=ts.volume[s],
    )


def rolling_log_score_backtest_conditioned(
    ts: StockReturnTS,
    calendar: EventCalendar,
    horizon_days: int = 5,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = 42,
    event_window: int = 0,
    name: Optional[str] = None,
) -> BacktestResult:
    """Event-conditioned sibling of rolling_log_score_backtest.

    At each holdout day t it (1) builds ConditionedReturnDistributions from the
    TRAINING window only (close[:t+1] -> returns[:t]) so there is no price
    look-ahead, and (2) computes the horizon event schedule from the full calendar
    (using KNOWN upcoming event dates — legitimate, since earnings dates are
    announced in advance), then forecasts with EventConditionedBootstrapForecaster.

    Per-iteration seed = base_seed + iteration decorrelates Monte-Carlo noise.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if holdout_days < 1:
        raise ValueError(f"holdout_days must be >= 1, got {holdout_days}")

    close = np.asarray(ts.close, dtype=float)
    dates = ts.dates
    n_bars = len(close)
    min_required = holdout_days + horizon_days + 2
    if n_bars < min_required:
        raise ValueError(f"need at least {min_required} bars, got {n_bars} for {ts.ticker}")

    T = n_bars - holdout_days - 1

    evaluations: list[ForecastEvaluation] = []
    for iteration, t in enumerate(range(T, n_bars - horizon_days)):
        spot = float(close[t])
        realized = float(close[t + horizon_days])

        # Distributions from training data only (no price look-ahead).
        ts_train = _truncate_ts(ts, t)
        conditioned = ConditionedReturnDistributions.build(
            ts_train, calendar, decay_lambda=decay_lambda, event_window=event_window
        )
        # Schedule from known future event dates over horizon returns [t, t+horizon).
        schedule = calendar.horizon_event_schedule(ts, start_idx=t, horizon_days=horizon_days)

        forecaster = EventConditionedBootstrapForecaster(
            conditioned, schedule, n_paths=n_paths, seed=base_seed + iteration
        )
        fcast = forecaster.forecast(horizon_days=horizon_days, spot=spot)

        evaluations.append(ForecastEvaluation(
            forecast_date=pd.Timestamp(dates[t]).date(),
            realized_date=pd.Timestamp(dates[t + horizon_days]).date(),
            spot=spot,
            realized=realized,
            forecast_mean=fcast.mean(),
            forecast_std=fcast.std(),
            percentile_of_realized=_weighted_cdf_at(fcast, realized),
            log_score=_log_density(fcast, realized),
        ))

    config = {
        "ticker": ts.ticker,
        "horizon_days": horizon_days,
        "holdout_days": holdout_days,
        "decay_lambda": decay_lambda,
        "n_paths": n_paths,
        "event_window": event_window,
        "n_events": len(calendar),
        "n_evaluations": len(evaluations),
    }
    label = name or f"{ts.ticker} h={horizon_days} event-conditioned"
    logger.info("rolling_log_score_backtest_conditioned %s: %d evaluations (%d events)",
                label, len(evaluations), len(calendar))
    return BacktestResult(name=label, evaluations=evaluations, config=config)


def rolling_log_score_backtest_garch_fhs_event(
    ts: StockReturnTS,
    calendar: EventCalendar,
    horizon_days: int = 5,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = 42,
    event_window: int = 0,
    name: Optional[str] = None,
) -> BacktestResult:
    """GARCH-FHS-event sibling of rolling_log_score_backtest_conditioned.

    At each holdout day t it builds a ReturnDistribution and
    ConditionedReturnDistributions from the TRAINING window only (no look-ahead),
    then forecasts with GarchFhsForecaster (event mode): GARCH-filtered vol on normal days,
    empirical event draws on event days.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if holdout_days < 1:
        raise ValueError(f"holdout_days must be >= 1, got {holdout_days}")

    from options_trader.forecast.garch_fhs_forecaster import GarchFhsForecaster

    close = np.asarray(ts.close, dtype=float)
    dates = ts.dates
    n_bars = len(close)
    min_required = holdout_days + horizon_days + 2
    if n_bars < min_required:
        raise ValueError(f"need at least {min_required} bars, got {n_bars} for {ts.ticker}")

    T = n_bars - holdout_days - 1

    evaluations: list[ForecastEvaluation] = []
    for iteration, t in enumerate(range(T, n_bars - horizon_days)):
        spot = float(close[t])
        realized = float(close[t + horizon_days])

        ts_train = _truncate_ts(ts, t)
        train_log_returns = np.log(ts_train.close[1:] / ts_train.close[:-1])
        rd_train = ReturnDistribution(train_log_returns, decay_lambda=decay_lambda, label=ts.ticker)

        conditioned = ConditionedReturnDistributions.build(
            ts_train, calendar, decay_lambda=decay_lambda, event_window=event_window
        )
        schedule = calendar.horizon_event_schedule(ts, start_idx=t, horizon_days=horizon_days)

        forecaster = GarchFhsForecaster(
            rd_train, n_paths=n_paths, seed=base_seed + iteration,
            conditioned=conditioned, event_schedule=schedule,
        )
        fcast = forecaster.forecast(horizon_days=horizon_days, spot=spot)

        evaluations.append(ForecastEvaluation(
            forecast_date=pd.Timestamp(dates[t]).date(),
            realized_date=pd.Timestamp(dates[t + horizon_days]).date(),
            spot=spot,
            realized=realized,
            forecast_mean=fcast.mean(),
            forecast_std=fcast.std(),
            percentile_of_realized=_weighted_cdf_at(fcast, realized),
            log_score=_log_density(fcast, realized),
        ))

    config = {
        "ticker": ts.ticker,
        "horizon_days": horizon_days,
        "holdout_days": holdout_days,
        "decay_lambda": decay_lambda,
        "n_paths": n_paths,
        "event_window": event_window,
        "n_events": len(calendar),
        "n_evaluations": len(evaluations),
    }
    label = name or f"{ts.ticker} h={horizon_days} garch-fhs-event"
    logger.info("rolling_log_score_backtest_garch_fhs_event %s: %d evaluations (%d events)",
                label, len(evaluations), len(calendar))
    return BacktestResult(name=label, evaluations=evaluations, config=config)
