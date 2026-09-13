"""
option_implied_backtest.py — backtest the Option-Implied (own-options) BENCHMARK.

This scores the option-implied forecaster (a ticker's OWN risk-neutral PDF, re-anchored
to spot — see forecast/option_implied_forecaster.py) on the rolling log-score backtest,
so it can be compared head-to-head against the production forecaster.

⚠️ This forecaster is a BENCHMARK, never a production default — it reproduces the
prices already for sale. The point of the comparison is the sign of the difference: if
our production forecaster scores HIGHER than this option-implied one, our view of S_T is
more accurate than the market's own implied view → predictive edge over the prices for
sale. (See the driver, scripts/run_option_implied_backtest.py.)

Per holdout day t (run_date = dates[t]):
  1. Build the ticker's own risk-neutral PDF AS OF run_date from that day's EOD option
     closes — reusing `build_eod_pdf` (liquid monthly expiry, OTM put+call,
     volume-weighted quadratic-in-log-moneyness smile; works for any symbol).
  2. Build `OptionImpliedForecaster` from it and forecast to close[t+horizon].
  3. Score (PIT + KDE log score) — identical scoring to `rolling_log_score_backtest`.

Dates where the PDF can't be built (illiquid chain, missing EOD bars) are SKIPPED and
counted, NOT silently degraded — a degrade would pollute the paired comparison.

Network-backed and slow (one chain fetch per run_date); `HistoricalTickerPdf` caches per
(ticker, run_date, horizon).
"""

from __future__ import annotations

import logging
from datetime import date
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
    build_eod_pdf,
    _spot_lookup_from_ts,
)
from options_trader.data.options_history import AlpacaOptionBarsProvider, OptionBarsProvider
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.breeden_litzenberger import DEFAULT_RISK_FREE_RATE, ImpliedPDF
from options_trader.forecast.option_implied_forecaster import OptionImpliedForecaster


logger = logging.getLogger(__name__)


class HistoricalTickerPdf:
    """Cached `(ticker, run_date, horizon) -> ImpliedPDF` from a ticker's OWN EOD bars.

    `spot_lookup(symbol, run_date)` supplies the underlying spot (e.g. the ticker's
    StockReturnTS close on run_date — see `_spot_lookup_from_ts`). Caches per
    (ticker, run_date, horizon) so repeated probes reuse the (expensive) chain fetch.
    Mirrors `option_implied_beta_backtest.HistoricalEodPdf`.
    """

    def __init__(self, spot_lookup, *, trading_client=None,
                 bars_provider: Optional[OptionBarsProvider] = None,
                 risk_free_rate: float = DEFAULT_RISK_FREE_RATE, **build_kwargs) -> None:
        self.spot_lookup = spot_lookup
        self.trading_client = trading_client
        self.bars_provider = bars_provider or AlpacaOptionBarsProvider()
        self.risk_free_rate = risk_free_rate
        self.build_kwargs = build_kwargs
        self._cache: dict[tuple[str, date, int], ImpliedPDF] = {}

    def __call__(self, symbol: str, run_date: date, horizon_days: int) -> ImpliedPDF:
        key = (symbol, run_date, horizon_days)
        if key not in self._cache:
            spot = float(self.spot_lookup(symbol, run_date))
            self._cache[key] = build_eod_pdf(
                symbol, run_date, horizon_days, spot,
                trading_client=self.trading_client, bars_provider=self.bars_provider,
                risk_free_rate=self.risk_free_rate, **self.build_kwargs,
            )
        return self._cache[key]


def rolling_log_score_backtest_option_implied(
    ts: StockReturnTS,
    ticker_pdf_fn,
    horizon_days: int = 21,
    holdout_days: int = 63,
    n_paths: int = 10_000,
    base_seed: int = 42,
    name: Optional[str] = None,
) -> BacktestResult:
    """Point-in-time rolling backtest of the option-implied (own-options) benchmark.

    `ticker_pdf_fn(symbol, run_date, horizon)` returns the as-of risk-neutral PDF for
    the ticker's own chain (use `HistoricalTickerPdf`). Dates where it raises are
    skipped and counted (never degraded), so the paired comparison stays clean.
    """
    if horizon_days < 1 or holdout_days < 1:
        raise ValueError("horizon_days and holdout_days must be >= 1")

    close = np.asarray(ts.close, dtype=float)
    dates = ts.dates
    n_bars = len(close)
    min_required = holdout_days + horizon_days + 2
    if n_bars < min_required:
        raise ValueError(f"need at least {min_required} bars, got {n_bars} for {ts.ticker}")

    T = n_bars - holdout_days - 1
    evaluations: list[ForecastEvaluation] = []
    skipped = 0

    for iteration, t in enumerate(range(T, n_bars - horizon_days)):
        run_date = pd.Timestamp(dates[t]).date()
        spot = float(close[t])
        realized = float(close[t + horizon_days])

        try:
            pdf = ticker_pdf_fn(ts.ticker, run_date, horizon_days)
            implied_returns = pdf.to_return_distribution()
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.info("%s @ %s: option PDF unavailable (%s); skipping", ts.ticker, run_date, exc)
            continue

        # _truncate_ts is not strictly required (the forecaster uses no ts history),
        # but kept for symmetry / future use and a no-look-ahead audit trail.
        _truncate_ts(ts, t)

        forecaster = OptionImpliedForecaster(
            implied_returns, n_paths=n_paths, seed=base_seed + iteration,
            pdf_horizon_days=horizon_days,
        )
        fcast = forecaster.forecast(horizon_days=horizon_days, spot=spot)

        evaluations.append(ForecastEvaluation(
            forecast_date=run_date,
            realized_date=pd.Timestamp(dates[t + horizon_days]).date(),
            spot=spot,
            realized=realized,
            forecast_mean=fcast.mean(),
            forecast_std=fcast.std(),
            percentile_of_realized=_weighted_cdf_at(fcast, realized),
            log_score=_log_density(fcast, realized),
        ))

    config = {
        "ticker": ts.ticker, "horizon_days": horizon_days, "holdout_days": holdout_days,
        "n_paths": n_paths, "n_evaluations": len(evaluations), "n_skipped": skipped,
    }
    label = name or f"{ts.ticker} h={horizon_days} option-implied"
    logger.info("option_implied backtest %s: %d evals, %d skipped",
                label, len(evaluations), skipped)
    return BacktestResult(name=label, evaluations=evaluations, config=config)


__all__ = [
    "HistoricalTickerPdf",
    "rolling_log_score_backtest_option_implied",
    "_spot_lookup_from_ts",
]
