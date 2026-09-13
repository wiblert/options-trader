"""
factories.py — build a forecaster for one ticker's LIVE forecast context.

The backtest selects forecasters through a factory (so it can score any
`Forecaster` subclass); production needs the same indirection so the
backtest-validated forecaster is the one that actually trades. A live forecaster
may need more than a `ReturnDistribution` — the event-conditioned one needs the
ticker's event calendar plus a FORWARD event schedule over the horizon — so the
factory takes a `ForecastContext` rather than a bare distribution.

    bootstrap_factory     -> BootstrapForecaster                  (baseline)
    EventBootstrapFactory -> EventConditionedBootstrapForecaster  (PRODUCTION DEFAULT, S16)
    garch_fhs_factory     -> GarchFhsForecaster (plain mode)       (backtest-accepted S11)
    GarchFhsFactory       -> GarchFhsForecaster (event-conditioned) (backtest-accepted S12;
                                                                      available, not the default)

Production default switched GarchFhsFactory -> EventBootstrapFactory (S16): the GARCH machinery added
only an insignificant +0.031 over event-bootstrap on the h=21 standardized check (p=0.38), so the
event conditioning — not the GARCH(1,1)/FHS layer — is the win. GarchFhsFactory stays available
(`--garch-fhs-events`). Both degrade gracefully (with a warning) to their plain forecaster if the event
calendar can't be built, so a flaky earnings feed never aborts a ticker.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from options_trader.data.events.calendar import EventCalendar
from options_trader.data.events.source import EventSource
from options_trader.data.stock_return_ts import StockReturnTS
from options_trader.forecast.base import Forecaster
from options_trader.forecast.bootstrap_forecaster import BootstrapForecaster
from options_trader.forecast.conditioned_returns import ConditionedReturnDistributions
from options_trader.forecast.event_bootstrap_forecaster import (
    EventConditionedBootstrapForecaster,
)
from options_trader.forecast.garch_fhs_forecaster import GarchFhsForecaster
from options_trader.forecast.beta import BetaCalculator, DEFAULT_BETA_WINDOW, align_log_returns
from options_trader.forecast.breeden_litzenberger import (
    DEFAULT_RISK_FREE_RATE,
    ImpliedPDF,
    implied_pdf_from_chain,
)
from options_trader.forecast.option_implied_beta_forecaster import (
    OptionImpliedBetaForecaster,
    idiosyncratic_residuals,
)
from options_trader.forecast.option_implied_forecaster import OptionImpliedForecaster
from options_trader.forecast.blend_forecaster import BlendForecaster
from options_trader.forecast.return_distribution import (
    DEFAULT_DECAY_LAMBDA,
    ReturnDistribution,
)


logger = logging.getLogger(__name__)

# How far past run_date to look for upcoming events when building the calendar.
# Generous so any earnings inside a held-to-expiry horizon is captured.
EVENT_LOOKAHEAD_DAYS = 400


@dataclass(frozen=True)
class ForecastContext:
    """Everything a live forecaster might need for one ticker, one run."""

    ticker: str
    ts: StockReturnTS            # full loaded history (training data)
    rd: ReturnDistribution       # log-returns of ts.close (built once by the caller)
    horizon: int                 # trading days to the selected expiry
    run_date: date
    n_paths: int
    seed: int


# A forecaster factory turns a per-ticker context into a ready-to-forecast Forecaster.
ForecasterFactory = Callable[[ForecastContext], Forecaster]


def bootstrap_factory(ctx: ForecastContext) -> Forecaster:
    """Plain iid bootstrap — the incumbent production default (no events, no network)."""
    return BootstrapForecaster(ctx.rd, n_paths=ctx.n_paths, seed=ctx.seed)


def garch_fhs_factory(ctx: ForecastContext) -> Forecaster:
    """GARCH(1,1)-filtered Historical Simulation — backtest-accepted (Session 11).

    Fits GARCH on the return history, filters to standardised residuals, then
    bootstraps those while propagating the variance recursion forward anchored on
    today's conditional variance.  Fixes the iid bootstrap's constant-vol blind spot.
    """
    return GarchFhsForecaster(ctx.rd, n_paths=ctx.n_paths, seed=ctx.seed)


class EventBootstrapFactory:
    """Event-conditioned bootstrap: routes the horizon days that carry a known
    upcoming event (e.g. earnings) to that event type's historical return
    distribution, normal days to the unconditional one.

    Backtest-accepted (Session 7: +0.032 mean log score, p=0.027 vs plain bootstrap; at h=21 the lift
    is far larger, +0.153). **PRODUCTION DEFAULT since S16** — the h=21 standardized check showed it
    captures ~83% of GARCH-FHS's lift with none of the GARCH machinery (the +0.031 GARCH adds on top is
    insignificant, p=0.38). The crucial live benefit: the forecast is no longer blind to an earnings
    print landing inside the option's horizon.
    """

    def __init__(self, event_source: EventSource, event_window: int = 0,
                 decay_lambda: float = DEFAULT_DECAY_LAMBDA) -> None:
        self.event_source = event_source
        self.event_window = event_window
        self.decay_lambda = decay_lambda

    def __call__(self, ctx: ForecastContext) -> Forecaster:
        data_start = pd.Timestamp(ctx.ts.dates[0]).date()
        cal_end = ctx.run_date + timedelta(days=EVENT_LOOKAHEAD_DAYS)
        try:
            calendar = EventCalendar.from_source(
                self.event_source, ctx.ticker, data_start, cal_end
            )
            conditioned = ConditionedReturnDistributions.build(
                ctx.ts, calendar, decay_lambda=self.decay_lambda,
                event_window=self.event_window,
            )
            schedule = calendar.forward_schedule(ctx.run_date, ctx.horizon)
        except Exception as exc:  # noqa: BLE001 — earnings feeds raise a zoo of types
            logger.warning(
                "%s: event calendar/schedule unavailable (%s); using plain bootstrap",
                ctx.ticker, exc,
            )
            return bootstrap_factory(ctx)

        n_event_days = sum(s is not None for s in schedule)
        if n_event_days:
            logger.info(
                "%s: %d event day(s) in the %d-day horizon: %s",
                ctx.ticker, n_event_days, ctx.horizon,
                [str(s) for s in schedule if s is not None],
            )
        return EventConditionedBootstrapForecaster(
            conditioned, schedule, n_paths=ctx.n_paths, seed=ctx.seed
        )


class GarchFhsFactory:
    """GARCH-FHS with event conditioning (accepted S12; available via --garch-fhs-events,
    NOT the default since S16 — see module docstring).

    Normal days use GARCH-filtered vol-state-conditioned draws; event days
    (earnings/FOMC) draw directly from the empirical event return distribution,
    then advance the GARCH variance state from the realised shock magnitude.

    Degrades to plain GarchFhsForecaster (no events) per-ticker if the event
    calendar is unavailable.
    """

    def __init__(self, event_source: EventSource, event_window: int = 0,
                 decay_lambda: float = DEFAULT_DECAY_LAMBDA) -> None:
        self.event_source = event_source
        self.event_window = event_window
        self.decay_lambda = decay_lambda

    def __call__(self, ctx: ForecastContext) -> Forecaster:
        data_start = pd.Timestamp(ctx.ts.dates[0]).date()
        cal_end = ctx.run_date + timedelta(days=EVENT_LOOKAHEAD_DAYS)
        try:
            calendar = EventCalendar.from_source(
                self.event_source, ctx.ticker, data_start, cal_end
            )
            conditioned = ConditionedReturnDistributions.build(
                ctx.ts, calendar, decay_lambda=self.decay_lambda,
                event_window=self.event_window,
            )
            schedule = calendar.forward_schedule(ctx.run_date, ctx.horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s: event calendar/schedule unavailable (%s); using plain GARCH-FHS",
                ctx.ticker, exc,
            )
            return garch_fhs_factory(ctx)

        n_event_days = sum(s is not None for s in schedule)
        if n_event_days:
            logger.info(
                "%s: %d event day(s) in the %d-day horizon: %s",
                ctx.ticker, n_event_days, ctx.horizon,
                [str(s) for s in schedule if s is not None],
            )
        return GarchFhsForecaster(
            ctx.rd, n_paths=ctx.n_paths, seed=ctx.seed,
            conditioned=conditioned, event_schedule=schedule,
        )


INDEX_SYMBOLS = ("SPY", "IWM")
# horizon trading days -> calendar days (≈ 7/5) to target an expiry near the horizon.
_TRADING_TO_CALENDAR = 7.0 / 5.0
# how far past run_date to look back for index histories used by beta/residuals.
_INDEX_HISTORY_YEARS = 3


# A function that returns the risk-neutral ImpliedPDF for one index, on one run.
IndexPdfFn = Callable[[str, date, int], ImpliedPDF]
# A function that returns a StockReturnTS of an index's history.
IndexHistoryFn = Callable[[str], StockReturnTS]


def live_eod_pdf(
    symbol: str,
    run_date: date,
    horizon_days: int,
    *,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> ImpliedPDF:
    """Production index-PDF source: live SPY/IWM call chain -> Breeden-Litzenberger.

    Picks the listed expiry closest to `run_date + horizon` (so the PDF's horizon
    matches the forecast), pulls that expiry's call smile, and derives the
    risk-neutral PDF. Network-backed — exercised live, stubbed in tests.
    """
    from options_trader.data.history import get_history, get_latest_price
    from options_trader.data.options_chain import get_option_chain
    from options_trader.valuation.option_valuer import OptionType

    spot = get_latest_price(symbol)
    if spot is None or spot <= 0:
        spot = float(get_history(symbol).close[-1])

    target = run_date + timedelta(days=max(1, round(horizon_days * _TRADING_TO_CALENDAR)))
    contracts = get_option_chain(
        symbol,
        option_type=OptionType.CALL,
        expiration_gte=target - timedelta(days=10),
        expiration_lte=target + timedelta(days=35),
    )
    expiries = sorted({c.expiry for c in contracts})
    if not expiries:
        raise ValueError(f"{symbol}: no listed call expiries near {target}")
    chosen = min(expiries, key=lambda e: abs((e - target).days))
    smile = [c for c in contracts if c.expiry == chosen]
    t_years = max((chosen - run_date).days, 1) / 365.0
    return implied_pdf_from_chain(smile, spot, t_years, risk_free_rate)


class LiveEodPdfCache:
    """Caches `live_eod_pdf` results per (symbol, run_date, horizon_days).

    `OptionImpliedBetaFactory` pulls the SPY/IWM PDF for EVERY ticker in a
    watchlist run; since run_date is shared and horizon is usually identical
    across names, most of those chain fetches are redundant. Share one instance
    across a `run_daily` call (via `index_pdf_fn=`) to fetch each distinct
    (symbol, run_date, horizon) combination only once.

    Single-flight per key: `run_daily` plans tickers concurrently (thread pool),
    so a naive check-then-fetch-then-store cache is defeated on a cold start —
    every thread misses before any of them finishes the (slow, network) fetch
    and writes back. Holding a PER-KEY lock across the whole miss (check + fetch
    + store) means the first thread to ask for a key fetches it once; every other
    thread asking for the SAME key blocks until that fetch lands, then reads the
    cached result instead of re-fetching. Different keys (e.g. SPY vs IWM) still
    fetch concurrently across threads.
    """

    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> None:
        self.risk_free_rate = risk_free_rate
        self._cache: dict[tuple[str, date, int], ImpliedPDF] = {}
        self._locks_guard = threading.Lock()
        self._key_locks: dict[tuple[str, date, int], threading.Lock] = {}

    def _lock_for(self, key: tuple[str, date, int]) -> threading.Lock:
        with self._locks_guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def __call__(self, symbol: str, run_date: date, horizon_days: int) -> ImpliedPDF:
        key = (symbol, run_date, horizon_days)
        with self._lock_for(key):
            if key not in self._cache:
                self._cache[key] = live_eod_pdf(
                    symbol, run_date, horizon_days, risk_free_rate=self.risk_free_rate
                )
            return self._cache[key]


class OptionImpliedBetaFactory:
    """Build the Option-Implied Beta forecaster for one ticker's live context.

    Extracts the SPY/IWM risk-neutral PDFs (via `index_pdf_fn`), computes the
    ticker's Beta vs each index and its idiosyncratic residual series (via
    `index_history_fn`), and assembles `OptionImpliedBetaForecaster`.

    NOT a production default — pending the log-score backtest (CLAUDE.md criterion).
    Degrades to plain bootstrap (with a warning) if any index input is unavailable,
    so a flaky index chain never aborts a ticker.

    Args:
        index_pdf_fn: (symbol, run_date, horizon) -> ImpliedPDF. Defaults to the
            live chain path; inject a stub in tests / a historical path in backtests.
        index_history_fn: symbol -> StockReturnTS for beta + residuals (default get_history).
        beta_window: rolling window for beta (trading days).
        blend_spy: weight on SPY in the 50/50 PDF + beta blend.
        risk_free_rate: passed to the default live PDF source.
    """

    def __init__(
        self,
        index_pdf_fn: Optional[IndexPdfFn] = None,
        index_history_fn: Optional[IndexHistoryFn] = None,
        beta_window: int = DEFAULT_BETA_WINDOW,
        blend_spy: float = 0.5,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        event_source: Optional[EventSource] = None,
        event_window: int = 0,
    ) -> None:
        self.index_pdf_fn = index_pdf_fn
        self.index_history_fn = index_history_fn
        self.beta_window = beta_window
        self.blend_spy = blend_spy
        self.risk_free_rate = risk_free_rate
        # When set, condition the IDIOSYNCRATIC residuals on single-name earnings
        # (matching the backtested event-OIB arm). MARKET events stay free via the
        # index PDF on the systematic side. None => plain OIB.
        self.event_source = event_source
        self.event_window = event_window

    def _pdf_fn(self) -> IndexPdfFn:
        if self.index_pdf_fn is not None:
            return self.index_pdf_fn
        return lambda sym, rd, h: live_eod_pdf(sym, rd, h, risk_free_rate=self.risk_free_rate)

    def _history_fn(self) -> IndexHistoryFn:
        if self.index_history_fn is not None:
            return self.index_history_fn
        from options_trader.data.history import get_history
        return lambda sym: get_history(sym)

    def __call__(self, ctx: ForecastContext) -> Forecaster:
        try:
            history_fn = self._history_fn()
            spy_ts = history_fn("SPY")
            iwm_ts = history_fn("IWM")

            pdf_fn = self._pdf_fn()
            spy_market = pdf_fn("SPY", ctx.run_date, ctx.horizon).to_return_distribution()
            iwm_market = pdf_fn("IWM", ctx.run_date, ctx.horizon).to_return_distribution()

            if self.event_source is not None:
                return self._build_event_oib(ctx, spy_ts, iwm_ts, spy_market, iwm_market)

            betas = BetaCalculator(self.beta_window).compute(ctx.ts, spy_ts, iwm_ts)
            blended_beta = betas.blended(self.blend_spy)

            # Idiosyncratic residuals over the aligned daily series.
            _dates, (s_ret, spy_ret, iwm_ret) = align_log_returns(ctx.ts, spy_ts, iwm_ts)
            market_blend = self.blend_spy * spy_ret + (1.0 - self.blend_spy) * iwm_ret
            eps = idiosyncratic_residuals(s_ret, market_blend, blended_beta)
            idio_rd = ReturnDistribution(eps)
        except Exception as exc:  # noqa: BLE001 — index feeds raise many types
            logger.warning(
                "%s: option-implied-beta inputs unavailable (%s); using plain bootstrap",
                ctx.ticker, exc,
            )
            return bootstrap_factory(ctx)

        logger.info(
            "%s: option-implied-beta β_spy=%.3f β_iwm=%.3f β_blend=%.3f",
            ctx.ticker, betas.beta_spy, betas.beta_iwm, blended_beta,
        )
        return OptionImpliedBetaForecaster(
            spy_market, iwm_market, betas.beta_spy, betas.beta_iwm, idio_rd,
            n_paths=ctx.n_paths, seed=ctx.seed,
            blend_spy=self.blend_spy, pdf_horizon_days=ctx.horizon,
        )

    def _build_event_oib(self, ctx, spy_ts, iwm_ts, spy_market, iwm_market) -> Forecaster:
        """Event-conditioned OIB (matches the backtested event-OIB arm).

        Partitions the ticker's idiosyncratic residuals by single-name earnings and
        routes the forward earnings days in the horizon to that distribution. Raises
        on feed failure (caller degrades to bootstrap). Lazy-imports the backtest
        helper to avoid a factories<->backtest import cycle.
        """
        from options_trader.backtest.option_implied_beta_backtest import (
            _conditioned_residuals,
            DEFAULT_IDIO_EVENT_TYPES,
        )

        data_start = pd.Timestamp(ctx.ts.dates[0]).date()
        cal_end = ctx.run_date + timedelta(days=EVENT_LOOKAHEAD_DAYS)
        calendar = EventCalendar.from_source(self.event_source, ctx.ticker, data_start, cal_end)
        beta_spy, beta_iwm, normal_rd, idio_event = _conditioned_residuals(
            ctx.ts, spy_ts, iwm_ts, calendar, blend_spy=self.blend_spy,
            beta_window=self.beta_window, event_window=self.event_window,
        )
        schedule = calendar.forward_schedule(ctx.run_date, ctx.horizon)
        schedule = [s if s in DEFAULT_IDIO_EVENT_TYPES else None for s in schedule]
        logger.info(
            "%s: event-OIB β_spy=%.3f β_iwm=%.3f event_days=%d",
            ctx.ticker, beta_spy, beta_iwm, sum(s is not None for s in schedule),
        )
        return OptionImpliedBetaForecaster(
            spy_market, iwm_market, beta_spy, beta_iwm, normal_rd,
            n_paths=ctx.n_paths, seed=ctx.seed, blend_spy=self.blend_spy,
            pdf_horizon_days=ctx.horizon, idio_event_dists=idio_event, event_schedule=schedule,
        )


class OptionImpliedFactory:
    """Build the Option-Implied (own-options) BENCHMARK forecaster for one ticker.

    Extracts the ticker's OWN risk-neutral PDF from its live option chain (via
    `pdf_fn`, defaulting to `live_eod_pdf`) and re-anchors it to spot. ⚠️ NEVER a
    production default — it reproduces the prices already for sale, so valuing
    those same options against it is circular. It exists only to benchmark the
    production forecaster's accuracy vs the market-implied PDF (see
    option_implied_forecaster.py). Degrades to plain bootstrap (with a warning) if the
    chain is unavailable, so a flaky feed never aborts a ticker.

    Args:
        pdf_fn: (symbol, run_date, horizon) -> ImpliedPDF. Defaults to the live chain
            path (`live_eod_pdf`); inject a stub in tests / a historical path in
            backtests.
        risk_free_rate: passed to the default live PDF source.
    """

    def __init__(
        self,
        pdf_fn: Optional[IndexPdfFn] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> None:
        self.pdf_fn = pdf_fn
        self.risk_free_rate = risk_free_rate

    def _pdf_fn(self) -> IndexPdfFn:
        if self.pdf_fn is not None:
            return self.pdf_fn
        return lambda sym, rd, h: live_eod_pdf(sym, rd, h, risk_free_rate=self.risk_free_rate)

    def __call__(self, ctx: ForecastContext) -> Forecaster:
        try:
            pdf = self._pdf_fn()(ctx.ticker, ctx.run_date, ctx.horizon)
            implied_returns = pdf.to_return_distribution()
        except Exception as exc:  # noqa: BLE001 — option feeds raise many types
            logger.warning(
                "%s: option-implied PDF unavailable (%s); using plain bootstrap",
                ctx.ticker, exc,
            )
            return bootstrap_factory(ctx)

        logger.info("%s: option-implied (own-options) benchmark forecaster built", ctx.ticker)
        return OptionImpliedForecaster(
            implied_returns, n_paths=ctx.n_paths, seed=ctx.seed,
            pdf_horizon_days=ctx.horizon,
        )


class BlendFactory:
    """Blend two (or more) factories' forecasters into a fixed-weight MIXTURE.

    The live mirror of `blend_forecaster` / `backtest.blend_backtest`: builds each
    sub-forecaster from the SAME `ForecastContext` and returns a `BlendForecaster`.
    Default 50/50. Each sub-factory already self-degrades to bootstrap on feed
    failure, so the blend is robust per-ticker.

    NOT a production default (pending the log-score backtest, CLAUDE.md criterion).
    The intended pairing is `EventBootstrapFactory` (historical/empirical) ⊕
    `OptionImpliedBetaFactory` (forward option-implied) — complementary information.

    Args:
        factories: the component factories (typically two).
        weights: blend weights (default equal). Renormalised.
    """

    def __init__(self, factories, weights=None) -> None:
        if len(factories) == 0:
            raise ValueError("need at least one factory to blend")
        self.factories = list(factories)
        self.weights = weights

    def __call__(self, ctx: ForecastContext) -> Forecaster:
        forecasters = [f(ctx) for f in self.factories]
        return BlendForecaster(forecasters, self.weights)


def event_days_in_horizon(forecaster: Forecaster) -> int:
    """Count horizon days routed to an event distribution, if the forecaster is
    event-conditioned (duck-typed on `event_schedule`); 0 otherwise."""
    schedule = getattr(forecaster, "event_schedule", None)
    if not schedule:
        return 0
    return sum(1 for s in schedule if s is not None)
