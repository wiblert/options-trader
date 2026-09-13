"""
blend_backtest.py — rolling backtest of the 50/50 blend (event-bootstrap ⊕ event-OIB).

Scores three arms IN LOCKSTEP per holdout day so they share forecast dates and skip the
same dates (clean paired comparisons):
  * eboot  — event-conditioned bootstrap (the production default)
  * oib    — event-conditioned Option-Implied Beta
  * blend  — the 50/50 probability mixture of the two terminal price distributions

The two component forecasters are built EXACTLY as their standalone backtests build them
(`rolling_log_score_backtest_conditioned` and
`rolling_log_score_backtest_option_implied_beta_event`), so the `eboot`/`oib` component
scores returned here are directly comparable to those standalone runs. The blend is then
the mixture of the two forecasts (no recomputation) via `blend_price_distributions`.

Dates where the SPY/IWM index PDF can't be built are skipped and counted for ALL THREE
arms (so the comparison is never contaminated by a partial/degraded date).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from options_trader.backtest.log_score import (
    BacktestResult,
    ForecastEvaluation,
    _log_density,
    _truncate_ts,
    _weighted_cdf_at,
)
from options_trader.backtest.option_implied_beta_backtest import (
    DEFAULT_IDIO_EVENT_TYPES,
    _conditioned_residuals,
    _truncate_ts_by_date,
)
from options_trader.data.events.calendar import EventCalendar
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.blend_forecaster import blend_price_distributions
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions
from options_trader.forecast.event_bootstrap_forecaster import EventConditionedBootstrapForecaster
from options_trader.forecast.option_implied_beta_forecaster import OptionImpliedBetaForecaster
from options_trader.forecast.return_distribution import DEFAULT_DECAY_LAMBDA


logger = logging.getLogger(__name__)


def _eval(forecast_date, realized_date, spot, realized, fcast) -> ForecastEvaluation:
    return ForecastEvaluation(
        forecast_date=forecast_date,
        realized_date=realized_date,
        spot=spot,
        realized=realized,
        forecast_mean=fcast.mean(),
        forecast_std=fcast.std(),
        percentile_of_realized=_weighted_cdf_at(fcast, realized),
        log_score=_log_density(fcast, realized),
    )


def rolling_log_score_backtest_blend_event(
    ts: StockReturnTS,
    spy_ts: StockReturnTS,
    iwm_ts: StockReturnTS,
    index_pdf_fn,
    calendar: EventCalendar,
    horizon_days: int = 21,
    holdout_days: int = 252,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = 42,
    beta_window: int = 252,
    event_window: int = 0,
    blend_weights=(0.5, 0.5),
    event_types=DEFAULT_IDIO_EVENT_TYPES,
    name: Optional[str] = None,
) -> dict[str, BacktestResult]:
    """Rolling backtest of event-bootstrap, event-OIB, and their 50/50 blend.

    Returns {"eboot": BacktestResult, "oib": BacktestResult, "blend": BacktestResult},
    all aligned (same forecast dates; same skipped dates). `index_pdf_fn(symbol,
    run_date, horizon)` returns the as-of SPY/IWM index PDF (use `HistoricalIndexPdf`).
    """
    if horizon_days < 1 or holdout_days < 1:
        raise ValueError("horizon_days and holdout_days must be >= 1")

    close = np.asarray(ts.close, dtype=float)
    dates = ts.dates
    n_bars = len(close)
    if n_bars < holdout_days + horizon_days + 2:
        raise ValueError(f"need >= {holdout_days + horizon_days + 2} bars, got {n_bars}")

    T = n_bars - holdout_days - 1
    ev = {"eboot": [], "oib": [], "blend": []}
    skipped = 0

    for iteration, t in enumerate(range(T, n_bars - horizon_days)):
        run_date = pd.Timestamp(dates[t]).date()
        realized_date = pd.Timestamp(dates[t + horizon_days]).date()
        spot = float(close[t])
        realized = float(close[t + horizon_days])
        seed = base_seed + iteration

        # Probe both index PDFs first; skip the date entirely if either is missing.
        try:
            spy_market = index_pdf_fn("SPY", run_date, horizon_days).to_return_distribution()
            iwm_market = index_pdf_fn("IWM", run_date, horizon_days).to_return_distribution()
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.info("%s @ %s: index PDF unavailable (%s); skipping", ts.ticker, run_date, exc)
            continue

        ts_tr = _truncate_ts(ts, t)

        # --- Arm 1: event-conditioned bootstrap (matches rolling_log_score_backtest_conditioned) ---
        conditioned = ConditionedReturnDistributions.build(
            ts_tr, calendar, decay_lambda=decay_lambda, event_window=event_window)
        schedule_eb = calendar.horizon_event_schedule(ts, start_idx=t, horizon_days=horizon_days)
        eboot = EventConditionedBootstrapForecaster(
            conditioned, schedule_eb, n_paths=n_paths, seed=seed)

        # --- Arm 2: event-conditioned OIB (matches rolling_log_score_backtest_option_implied_beta_event) ---
        spy_tr = _truncate_ts_by_date(spy_ts, run_date)
        iwm_tr = _truncate_ts_by_date(iwm_ts, run_date)
        beta_spy, beta_iwm, normal_rd, idio_event = _conditioned_residuals(
            ts_tr, spy_tr, iwm_tr, calendar, beta_window=beta_window,
            decay_lambda=decay_lambda, event_window=event_window, event_types=event_types)
        schedule_oib = calendar.forward_schedule(run_date, horizon_days)
        schedule_oib = [s if s in event_types else None for s in schedule_oib]
        oib = OptionImpliedBetaForecaster(
            spy_market, iwm_market, beta_spy, beta_iwm, normal_rd,
            n_paths=n_paths, seed=seed, pdf_horizon_days=horizon_days,
            idio_event_dists=idio_event, event_schedule=schedule_oib)

        # Forecast each once, then blend the two terminal distributions (the mixture).
        pd_eb = eboot.forecast(horizon_days=horizon_days, spot=spot)
        pd_oib = oib.forecast(horizon_days=horizon_days, spot=spot)
        pd_blend = blend_price_distributions([pd_eb, pd_oib], blend_weights)

        ev["eboot"].append(_eval(run_date, realized_date, spot, realized, pd_eb))
        ev["oib"].append(_eval(run_date, realized_date, spot, realized, pd_oib))
        ev["blend"].append(_eval(run_date, realized_date, spot, realized, pd_blend))

    base_label = name or f"{ts.ticker} h={horizon_days}"
    config_common = {
        "ticker": ts.ticker, "horizon_days": horizon_days, "holdout_days": holdout_days,
        "n_paths": n_paths, "beta_window": beta_window, "blend_weights": tuple(blend_weights),
        "n_skipped": skipped, "n_events": len(calendar),
    }
    out = {}
    for arm in ("eboot", "oib", "blend"):
        cfg = dict(config_common, n_evaluations=len(ev[arm]))
        out[arm] = BacktestResult(name=f"{base_label} {arm}", evaluations=ev[arm], config=cfg)
    logger.info("blend backtest %s: %d evals/arm, %d skipped", base_label, len(ev["blend"]), skipped)
    return out
