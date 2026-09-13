"""
option_implied_beta_backtest.py — backtest the Option-Implied Beta forecaster.

This wires the historical index PDF (component 1: expired SPY/IWM EOD option chains →
Breeden-Litzenberger) into a rolling, point-in-time log-score backtest so the
forecaster can be gated by the standing decision criterion (CLAUDE.md).

Per holdout day t (run_date = dates[t]):
  1. Build the SPY & IWM risk-neutral PDFs AS OF run_date from that day's EOD option
     closes (a clean OTM-put + OTM-call, volume-weighted, quadratic-in-log-moneyness
     smile — see `build_eod_pdf` and docs/sessions.md S16 for why).
  2. Truncate the ticker AND index histories to <= run_date (no look-ahead) and let
     `OptionImpliedBetaFactory` assemble the forecaster (betas + idiosyncratic
     residuals from training data, index PDFs from step 1).
  3. Forecast to close[t+horizon] and score (PIT + KDE log score) — identical
     scoring to `rolling_log_score_backtest`, so results are directly comparable.

Dates where the index PDF can't be built (illiquid expiry, missing bars) are SKIPPED
and counted, NOT silently degraded to bootstrap — a degrade would pollute the paired
comparison against the incumbent.

Network-backed and slow (two index chains per run_date); `HistoricalEodPdf` caches
per (symbol, run_date, horizon) so multiple tickers sharing a calendar reuse PDFs.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from datetime import date, datetime, timedelta, timezone
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
from options_trader.data.options_history import (
    AlpacaOptionBarsProvider,
    OptionBarsProvider,
    list_contracts,
)
from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.event import EventType
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.beta import compute_beta
from options_trader.forecast.breeden_litzenberger import (
    DEFAULT_RISK_FREE_RATE,
    ImpliedPDF,
    MIN_SMILE_POINTS,
    implied_pdf_from_iv,
)
from options_trader.forecast.factories import ForecastContext, OptionImpliedBetaFactory
from options_trader.forecast.option_implied_beta_forecaster import OptionImpliedBetaForecaster
from options_trader.forecast.return_distribution import (
    DEFAULT_DECAY_LAMBDA,
    ReturnDistribution,
)
from options_trader.valuation.black_scholes import implied_vol


# Single-name event types OIB conditions its IDIOSYNCRATIC residuals on. MARKET
# events (FOMC, CPI) are deliberately excluded — they are already priced into the
# index PDF on the systematic side, so conditioning here too would double-count.
DEFAULT_IDIO_EVENT_TYPES = (EventType.EARNINGS,)


logger = logging.getLogger(__name__)


# Defaults for the EOD index smile.
DEFAULT_STRIKE_PCT = 0.20      # strike window spot·(1±pct)
DEFAULT_MIN_VOLUME = 20.0      # drop stale/illiquid last-trade strikes
DEFAULT_MAX_PER_SIDE = 60      # cap strikes per wing (bars-request economy)
_TRADING_TO_CALENDAR = 7.0 / 5.0


def _eod_close_by_strike(
    refs, run_date: date, bars_provider: OptionBarsProvider
) -> dict[float, tuple[float, float]]:
    """{strike: (close, volume)} for the bars dated EXACTLY run_date.

    Exact-date only: a "most recent <= run_date" fallback mixes prices from
    different days into one smile and wrecks the second derivative.
    """
    by_symbol = {r.symbol: r.strike for r in refs}
    bars = bars_provider.get_eod_bars(
        list(by_symbol), run_date - timedelta(days=4), run_date + timedelta(days=1)
    )
    out: dict[float, tuple[float, float]] = {}
    for symbol, blist in bars.items():
        for b in blist:
            if b.bar_date == run_date and b.close > 0:
                out[by_symbol[symbol]] = (b.close, b.volume)
                break
    return out


def build_eod_pdf(
    symbol: str,
    run_date: date,
    horizon_days: int,
    spot: float,
    *,
    trading_client=None,
    bars_provider: Optional[OptionBarsProvider] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    strike_pct: float = DEFAULT_STRIKE_PCT,
    min_volume: float = DEFAULT_MIN_VOLUME,
    max_per_side: int = DEFAULT_MAX_PER_SIDE,
) -> ImpliedPDF:
    """Risk-neutral PDF for an index AS OF a past run_date, from EOD option bars.

    Discovers the liquid MONTHLY expiry near run_date+horizon, takes OTM puts
    (K<spot) + OTM calls (K>=spot), converts EOD closes to IV, and fits a
    volume-weighted quadratic-in-log-moneyness smile (Breeden-Litzenberger).

    Raises ValueError if no liquid expiry / too few priced OTM strikes.
    """
    if bars_provider is None:
        bars_provider = AlpacaOptionBarsProvider()

    target = run_date + timedelta(days=max(1, round(horizon_days * _TRADING_TO_CALENDAR)))
    lo, hi = spot * (1 - strike_pct), spot * (1 + strike_pct)

    calls = list_contracts(symbol, expiration_gte=target - timedelta(days=10),
                           expiration_lte=target + timedelta(days=14),
                           option_type="call", status="inactive",
                           strike_gte=lo, strike_lte=hi, trading_client=trading_client)
    puts = list_contracts(symbol, expiration_gte=target - timedelta(days=10),
                          expiration_lte=target + timedelta(days=14),
                          option_type="put", status="inactive",
                          strike_gte=lo, strike_lte=hi, trading_client=trading_client)
    if not calls:
        raise ValueError(f"{symbol}: no expired call contracts near {target}")

    # Pick the most liquid (max contract count) expiry among the closest few to target —
    # reliably the monthly; weeklies often have no historical daily bars.
    counts = Counter(c.expiry for c in calls)
    candidates = sorted(counts, key=lambda e: abs((e - target).days))[:5]
    chosen = max(candidates, key=lambda e: counts[e])
    t_years = max((chosen - run_date).days, 1) / 365.0

    call_refs = sorted((c for c in calls if c.expiry == chosen), key=lambda c: c.strike)
    put_refs = sorted((p for p in puts if p.expiry == chosen), key=lambda c: c.strike)
    # OTM wings, capped per side for bars-request economy (evenly subsampled).
    otm_calls = [c for c in call_refs if c.strike >= spot]
    otm_puts = [p for p in put_refs if p.strike < spot]

    def _subsample(refs):
        if len(refs) <= max_per_side:
            return refs
        idx = np.linspace(0, len(refs) - 1, max_per_side).round().astype(int)
        return [refs[i] for i in sorted(set(idx))]

    call_px = _eod_close_by_strike(_subsample(otm_calls), run_date, bars_provider)
    put_px = _eod_close_by_strike(_subsample(otm_puts), run_date, bars_provider)

    strikes, ivs, vols = [], [], []
    for K, (px, vol) in put_px.items():
        if vol < min_volume:
            continue
        try:
            ivs.append(implied_vol(px, spot, K, t_years, risk_free_rate, "put"))
            strikes.append(K); vols.append(vol)
        except ValueError:
            pass
    for K, (px, vol) in call_px.items():
        if vol < min_volume:
            continue
        try:
            ivs.append(implied_vol(px, spot, K, t_years, risk_free_rate, "call"))
            strikes.append(K); vols.append(vol)
        except ValueError:
            pass

    if len(strikes) < MIN_SMILE_POINTS:
        raise ValueError(
            f"{symbol} @ {run_date}: only {len(strikes)} liquid OTM strikes for "
            f"expiry {chosen} (need >= {MIN_SMILE_POINTS})"
        )

    pdf = implied_pdf_from_iv(strikes, ivs, spot, t_years, risk_free_rate,
                              weights=vols, method="quadratic")
    logger.debug("%s EOD PDF @ %s: expiry %s T=%.3f %d OTM strikes mean=%.2f fwd=%.2f",
                 symbol, run_date, chosen, t_years, len(strikes), pdf.mean_price(),
                 spot * np.exp(risk_free_rate * t_years))
    return pdf


class HistoricalEodPdf:
    """Cached `IndexPdfFn`: (symbol, run_date, horizon) -> ImpliedPDF from EOD bars.

    `spot_lookup(symbol, run_date)` supplies the index spot (e.g. the index
    StockReturnTS close on run_date). Caches per (symbol, run_date, horizon) so
    multiple tickers sharing a run_date reuse the (expensive) chain fetch.
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


def _spot_lookup_from_ts(*series: StockReturnTS):
    """Build a spot_lookup(symbol, run_date) -> close on-or-before run_date."""
    frames = {ts.ticker: pd.Series(np.asarray(ts.close, float),
                                   index=pd.DatetimeIndex(ts.dates)) for ts in series}

    def lookup(symbol: str, run_date: date) -> float:
        s = frames[symbol]
        s = s[s.index <= pd.Timestamp(run_date)]
        if s.empty:
            raise ValueError(f"{symbol}: no close on/before {run_date}")
        return float(s.iloc[-1])

    return lookup


def rolling_log_score_backtest_option_implied_beta(
    ts: StockReturnTS,
    spy_ts: StockReturnTS,
    iwm_ts: StockReturnTS,
    index_pdf_fn,
    horizon_days: int = 21,
    holdout_days: int = 21,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = 42,
    beta_window: int = 252,
    name: Optional[str] = None,
) -> BacktestResult:
    """Point-in-time rolling backtest of the Option-Implied Beta forecaster.

    `index_pdf_fn(symbol, run_date, horizon)` returns the as-of index PDF (use
    `HistoricalEodPdf`). Dates where it raises are skipped and counted.
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

        # Probe the index PDFs first; skip the date if either is unavailable
        # (so the comparison is never contaminated by a bootstrap fallback).
        try:
            index_pdf_fn("SPY", run_date, horizon_days)
            index_pdf_fn("IWM", run_date, horizon_days)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.info("%s @ %s: index PDF unavailable (%s); skipping", ts.ticker, run_date, exc)
            continue

        ts_train = _truncate_ts(ts, t)
        rd_train = ReturnDistribution(np.diff(np.log(ts_train.close)), decay_lambda=decay_lambda, label=ts.ticker)
        spy_train = _truncate_ts_by_date(spy_ts, run_date)
        iwm_train = _truncate_ts_by_date(iwm_ts, run_date)

        ctx = ForecastContext(
            ticker=ts.ticker, ts=ts_train, rd=rd_train, horizon=horizon_days,
            run_date=run_date, n_paths=n_paths, seed=base_seed + iteration,
        )
        factory = OptionImpliedBetaFactory(
            index_pdf_fn=index_pdf_fn,
            index_history_fn=lambda s, _m={"SPY": spy_train, "IWM": iwm_train}: _m[s],
            beta_window=beta_window,
        )
        forecaster = factory(ctx)
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
        "decay_lambda": decay_lambda, "n_paths": n_paths, "beta_window": beta_window,
        "n_evaluations": len(evaluations), "n_skipped": skipped,
    }
    label = name or f"{ts.ticker} h={horizon_days} option-implied-beta"
    logger.info("option_implied_beta backtest %s: %d evals, %d skipped",
                label, len(evaluations), skipped)
    return BacktestResult(name=label, evaluations=evaluations, config=config)


def _conditioned_residuals(
    ts: StockReturnTS,
    spy_ts: StockReturnTS,
    iwm_ts: StockReturnTS,
    calendar: EventCalendar,
    *,
    blend_spy: float = 0.5,
    beta_window: int = 252,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    event_window: int = 0,
    event_types=DEFAULT_IDIO_EVENT_TYPES,
):
    """Betas + event-partitioned IDIOSYNCRATIC residuals, all indexed to `ts`.

    SPY/IWM are reindexed onto `ts`'s trading dates (they share the NYSE calendar,
    so this drops ~nothing) so the residual series lines up 1:1 with the calendar's
    return indices. Residuals on `event_types` days (default earnings) become the
    event-conditioned idiosyncratic distribution; the rest are `normal`.

    Returns (beta_spy, beta_iwm, normal_rd, {EventType: event_rd}).
    """
    ridx = pd.DatetimeIndex(ts.dates)
    def _reindexed_returns(src: StockReturnTS) -> np.ndarray:
        s = pd.Series(np.asarray(src.close, float), index=pd.DatetimeIndex(src.dates))
        c = s.reindex(ridx).ffill().to_numpy()
        return np.diff(np.log(c))

    r_stock = np.diff(np.log(np.asarray(ts.close, float)))
    r_spy = _reindexed_returns(spy_ts)
    r_iwm = _reindexed_returns(iwm_ts)
    finite = np.isfinite(r_stock) & np.isfinite(r_spy) & np.isfinite(r_iwm)

    beta_spy = compute_beta(r_stock[finite], r_spy[finite], beta_window)
    beta_iwm = compute_beta(r_stock[finite], r_iwm[finite], beta_window)
    blended_beta = blend_spy * beta_spy + (1.0 - blend_spy) * beta_iwm

    market_blend = blend_spy * r_spy + (1.0 - blend_spy) * r_iwm
    eps = r_stock - blended_beta * market_blend  # length n, indexed like the returns
    n = len(eps)

    idx_by_type = calendar.event_return_indices(ts)
    event_mask = np.zeros(n, dtype=bool)
    by_type: dict = {}
    for etype, idxs in idx_by_type.items():
        if etype not in event_types:
            continue
        if event_window > 0:
            idxs = np.unique(np.concatenate([
                np.clip(idxs + d, 0, n - 1) for d in range(-event_window, event_window + 1)
            ]))
        idxs = idxs[finite[idxs]]
        event_mask[idxs] = True
        samples = eps[idxs]
        if len(samples) >= 2:  # need >= 2 for a usable distribution
            by_type[etype] = ReturnDistribution(samples, decay_lambda=decay_lambda, label=ts.ticker)

    normal = eps[finite & ~event_mask]
    if len(normal) == 0:
        normal = eps[finite]
    normal_rd = ReturnDistribution(normal, decay_lambda=decay_lambda, label=ts.ticker)
    return beta_spy, beta_iwm, normal_rd, by_type


def rolling_log_score_backtest_option_implied_beta_event(
    ts: StockReturnTS,
    spy_ts: StockReturnTS,
    iwm_ts: StockReturnTS,
    index_pdf_fn,
    calendar: EventCalendar,
    horizon_days: int = 21,
    holdout_days: int = 63,
    decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    n_paths: int = 10_000,
    base_seed: int = 42,
    beta_window: int = 252,
    event_window: int = 0,
    event_types=DEFAULT_IDIO_EVENT_TYPES,
    name: Optional[str] = None,
) -> BacktestResult:
    """Event-conditioned OIB backtest: systematic from index options (carries MARKET
    events for free), idiosyncratic residuals conditioned on single-name EARNINGS.

    Per holdout day, the idiosyncratic residual distribution is partitioned from
    TRAINING data only; the forward earnings schedule over the horizon uses known
    upcoming earnings dates. Dates with no index PDF are skipped and counted.
    """
    close = np.asarray(ts.close, dtype=float)
    dates = ts.dates
    n_bars = len(close)
    if n_bars < holdout_days + horizon_days + 2:
        raise ValueError(f"need >= {holdout_days + horizon_days + 2} bars, got {n_bars}")

    T = n_bars - holdout_days - 1
    evaluations: list[ForecastEvaluation] = []
    skipped = 0

    for iteration, t in enumerate(range(T, n_bars - horizon_days)):
        run_date = pd.Timestamp(dates[t]).date()
        spot = float(close[t])
        realized = float(close[t + horizon_days])
        try:
            spy_market = index_pdf_fn("SPY", run_date, horizon_days).to_return_distribution()
            iwm_market = index_pdf_fn("IWM", run_date, horizon_days).to_return_distribution()
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.info("%s @ %s: index PDF unavailable (%s); skipping", ts.ticker, run_date, exc)
            continue

        ts_tr = _truncate_ts(ts, t)
        spy_tr = _truncate_ts_by_date(spy_ts, run_date)
        iwm_tr = _truncate_ts_by_date(iwm_ts, run_date)
        beta_spy, beta_iwm, normal_rd, idio_event = _conditioned_residuals(
            ts_tr, spy_tr, iwm_tr, calendar, beta_window=beta_window,
            decay_lambda=decay_lambda, event_window=event_window, event_types=event_types)

        # Forward earnings schedule over the horizon (single-name events only).
        schedule = calendar.forward_schedule(run_date, horizon_days)
        schedule = [s if s in event_types else None for s in schedule]

        forecaster = OptionImpliedBetaForecaster(
            spy_market, iwm_market, beta_spy, beta_iwm, normal_rd,
            n_paths=n_paths, seed=base_seed + iteration, pdf_horizon_days=horizon_days,
            idio_event_dists=idio_event, event_schedule=schedule)
        fcast = forecaster.forecast(horizon_days=horizon_days, spot=spot)

        evaluations.append(ForecastEvaluation(
            forecast_date=run_date,
            realized_date=pd.Timestamp(dates[t + horizon_days]).date(),
            spot=spot, realized=realized,
            forecast_mean=fcast.mean(), forecast_std=fcast.std(),
            percentile_of_realized=_weighted_cdf_at(fcast, realized),
            log_score=_log_density(fcast, realized)))

    config = {"ticker": ts.ticker, "horizon_days": horizon_days, "holdout_days": holdout_days,
              "n_paths": n_paths, "beta_window": beta_window, "n_evaluations": len(evaluations),
              "n_skipped": skipped, "n_events": len(calendar)}
    label = name or f"{ts.ticker} h={horizon_days} option-implied-beta-event"
    logger.info("option_implied_beta_event %s: %d evals, %d skipped", label, len(evaluations), skipped)
    return BacktestResult(name=label, evaluations=evaluations, config=config)


def _truncate_ts_by_date(ts: StockReturnTS, run_date: date) -> StockReturnTS:
    """Keep bars with date <= run_date (point-in-time index history)."""
    mask = pd.DatetimeIndex(ts.dates) <= pd.Timestamp(run_date)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"{ts.ticker}: no bars on/before {run_date}")
    last = idx[-1] + 1
    return dataclasses.replace(
        ts, dates=ts.dates[:last], open=ts.open[:last], high=ts.high[:last],
        low=ts.low[:last], close=ts.close[:last], volume=ts.volume[:last],
    )
