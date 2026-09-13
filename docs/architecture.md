# Architecture

Current-state component map of the forecaster pipeline. (History of how it got here:
`docs/sessions.md`. Why decisions were made: `docs/decisions.md`.)

Pipeline: **history → return distribution(s) → forecaster → price distribution → valuation → Kelly sizing → portfolio caps → broker**, with the forecaster layer evaluated by the **log-score backtest**. High-level/holistic view + improvement backlog: `docs/architecture-overview.md`. Operating it + change procedure: `docs/runbook.md`.

---

## data/ — ingestion + events

**`history.py`** — `get_history(ticker, start=None, end=None) -> StockReturnTS`. Alpaca primary
(`Adjustment.ALL`, split/div-adjusted), yfinance fallback. 3-year default lookback. Typed errors:
`CredentialsError` (propagates, never silent-demotes) vs `DataUnavailableError`. 30s timeouts.

**`stock_return_ts.py`** — `StockReturnTS` dataclass: `ticker`, `dates` (datetime64[ns], US/Eastern
tz-naive midnight, sorted ascending), `open/high/low/close/volume` (parallel float64),
`source/start/end`. Aligned-array invariant in `__post_init__` vs `OHLCV_FIELDS`.

**`options_chain.py`** — `get_option_chain(underlying, *, option_type=None, expiration_gte/lte,
strike_gte/lte, feed="indicative", include_untradable=False) -> list[OptionContract]`. Merges two
Alpaca calls on the contract symbol: `TradingClient.get_option_contracts` (tradable master list +
open interest, paginated, `paper=True`) and `OptionHistoricalDataClient.get_option_chain` (latest
bid/ask + IV; greeks ignored). Tradable-only by default; quotes best-effort (snapshot failure →
quote-less contracts). Mirrors `history.py`: `CredentialsError` propagates (reused from `history`),
worker-thread timeout, typed `OptionsDataUnavailableError`. Pure `_build_contracts(raw, snapshots,
include_untradable)` is the testable merge core. Emits the valuation-layer `OptionContract`.

**`options_history.py`** — historical EOD option bars for BACKTESTING the option-implied PDF (live
runs use `options_chain.py` snapshots instead). `OptionBarsProvider` Protocol (swappable Alpaca→
Databento) + `AlpacaOptionBarsProvider` (`OptionHistoricalDataClient.get_option_bars`, `TimeFrame.Day`,
injectable client). Pure `_bars_from_alpaca_df` parse core; `OptionBar` dataclass. Mirrors
`options_chain.py`: `CredentialsError` propagates, worker-thread timeout, typed
`OptionsHistoryUnavailableError`. `get_option_bars_eod(symbols, start, end, provider=None)`.

**`events/`** — the event layer (event conditioning):
- `event.py` — `Event` (frozen dataclass: ticker, tz-aware `timestamp`, `event_type`, `timing`,
  `metadata`) + `EventType` / `EventTiming` str-enums. `timing` (BMO/AMC) is first-class because it
  decides which return the move attaches to.
- `source.py` — `EventSource` Protocol (`fetch_events(ticker, start, end) -> list[Event]`, `end`
  may be future) + `EventSourceError`.
- `yfinance_source.py` — `YFinanceEarningsSource` (`get_earnings_dates`, historical + upcoming;
  pure `_rows_to_events` helper for testing).
- `fomc_source.py` — `FomcCalendarSource`: scrapes the Fed's official FOMC calendar page
  (~6 years: recent history + scheduled future, so it feeds both conditioning AND the forward
  schedule). Auto-updating (replaces stale hand-entered FOMC rows). Macro → tagged to the queried
  ticker. Pure `_parse_fomc_calendar(html)` core (handles cross-month meetings, SEP `*` markers,
  excludes `notation vote`). Timing default BMO = announcement-day's own return (statement ~2pm ET).
- `manual_calendar.py` — `ManualCalendarSource(csv)` reads `config/manual_events.csv`
  (`ticker,date,event_type,timing,note`; `*` ticker = all). For ad-hoc / other macro / FDA
  (FOMC now comes from `fomc_source`).
- `composite.py` — `CompositeEventSource(*sources)` merge + dedup (ticker, date, type).
- `calendar.py` — `EventCalendar(ticker, events)`; `from_source(...)`. **Load-bearing mapping** below.

### Load-bearing: event → return-index mapping (`calendar.py`)
Return index `i` = move `close[i]→close[i+1]` (the return OF day `i+1`). For event date `d`,
`q = searchsorted(dates, d)`:
- trading-day **AMC** (or UNKNOWN) → index `q` (next session's return — the META after-close case)
- trading-day **BMO** → index `q-1` (same day's return)
- holiday/weekend → index `q-1` (next available session's return)
- out-of-range `[0, n-2]` dropped. Collision precedence: EARNINGS > FDA > MACRO_* > DIVIDEND_EX.

`event_return_indices(ts) -> {EventType: int[]}` (historical partition).
`horizon_event_schedule(ts, start_idx, horizon) -> list[EventType|None]` (per-horizon-day labels;
horizon return indices `[start_idx, start_idx+horizon)`) — the IN-HISTORY path (backtest).
`forward_schedule(run_date, horizon) -> list[EventType|None]` — the LIVE path: projects the next K
trading days (none exist in `ts` yet) and maps KNOWN upcoming events onto them via the same BMO/AMC
geometry (laid on a synthetic `[run_date, F0, F1, …]` axis so the return index == horizon day).

---

## forecast/ — distributions + forecasters

**`return_distribution.py`** — `ReturnDistribution`: weighted empirical distribution over daily
log-returns. `.samples` (time-ordered oldest→newest), `.weights` (decay, λ=0.99 default).
`add_sample` (exact incremental update via `_raw_weights`), `smooth_samples` (KDE/Silverman jitter),
`from_normal_distribution`. **PURE data container — never holds sampling strategy or event logic.**

**`price_distribution.py`** — `PriceDistribution` over terminal prices. `mean/std/quantile`. Output
type of every `forecast()`.

**`base.py`** — `Forecaster` ABC: `forecast(horizon_days, spot) -> PriceDistribution`.

**Forecasters** (uniform constructor `(distribution, n_paths, seed)` unless noted):
- `bootstrap_forecaster.py:BootstrapForecaster` — draws `(n_paths, K)` iid daily returns, sums,
  `spot*exp`. The baseline; degrade target for every event-aware factory.
- `block_bootstrap_forecaster.py:BlockBootstrapForecaster` — one contiguous K-day block per path
  (overlapping windows via cumsum-diff, end-anchored weights). Rejected unconditionally; kept for
  event-conditioned use.
- `gaussian_forecaster.py:GaussianForecaster` — BS baseline: N(K·μ, K·σ²) from weighted moments.
- `event_bootstrap_forecaster.py:EventConditionedBootstrapForecaster` — routes each horizon day's
  draw to `conditioned.distribution_for(schedule[k])`; schedule + distributions bound at construction.
- `garch_fhs_forecaster.py:GarchFhsForecaster` — GARCH(1,1) variance-targeting QMLE → filter to
  standardized residuals → bootstrap + propagate σ² forward (vol-state conditional). Event params are
  keyword-only optionals (normal days GARCH-filtered, event days draw the empirical event return).
  Backtest-accepted (S11/S12); available via `--garch-fhs` / `--garch-fhs-events`.
- `option_implied_beta_forecaster.py:OptionImpliedBetaForecaster` — index (SPY/IWM) risk-neutral PDF ·
  Beta + summed-daily idiosyncratic residual; optional event-conditioned idiosyncratic (`idio_event_
  dists`+`event_schedule`). One systematic draw per path (PDF is for the expiry). See its components below.
- `option_implied_forecaster.py:OptionImpliedForecaster` — re-anchors a ticker's OWN option-implied PDF
  to spot (deterministic; weighted samples carried through). **BENCHMARK ONLY, never a default** —
  reproduces the prices for sale (circular to trade). See `docs/decisions.md`.
- `blend_forecaster.py:BlendForecaster` + `blend_price_distributions()` — fixed-weight probability
  MIXTURE of N sub-forecasters' `PriceDistribution`s (concatenate prices, scale each side's weights →
  still sums to 1; preserves each shape). **The 50/50 event-bootstrap ⊕ event-OIB blend is the
  PRODUCTION DEFAULT (S18).** Constructor takes forecasters + weights, not `(distribution, n_paths, seed)`.

**`conditioned_returns.py`** — `ConditionedReturnDistributions`: `normal` + `{EventType:
ReturnDistribution}`, built by partitioning the return series via an `EventCalendar`
(`event_window` widens each event ±N). The bridge between events and the forecaster.
`ReturnDistribution` is constructed from index subsets — never modified.

**`factories.py`** — `ForecastContext` + `ForecasterFactory` build the per-ticker forecaster for the
LIVE path (production mirror of the backtest's factory). Each builds its forecaster from `ctx` and
degrades to `bootstrap_factory` if its feed fails. Factories:
- `bootstrap_factory` / `garch_fhs_factory` — plain baselines.
- `EventBootstrapFactory` — `EventCalendar` from a source → conditioned distributions → `forward_schedule`
  → `EventConditionedBootstrapForecaster`.
- `GarchFhsFactory` — event-conditioned GARCH-FHS (available via `--garch-fhs-events`).
- `OptionImpliedBetaFactory` — injectable index-PDF (`live_index_pdf`, SPY/IWM chain→BL) + index-history
  fns → betas + idiosyncratic residuals → `OptionImpliedBetaForecaster`. **Optional `event_source`**
  (S18): when set, conditions the idiosyncratic residuals on earnings (reuses `_conditioned_residuals`
  via lazy import) so the live OIB matches the backtested event-OIB.
- `OptionImpliedFactory` — own-options benchmark (NOT a default).
- `BlendFactory(factories, weights)` — builds each sub-factory's forecaster and wraps in
  `BlendForecaster`. **`_default_forecaster_factory` returns `BlendFactory([EventBootstrapFactory,
  OptionImpliedBetaFactory(event_source=…)], 0.5/0.5)` — the production default (S18).**
`event_days_in_horizon()` reads a forecaster's `event_schedule` back (BlendForecaster exposes the first
sub-forecaster's schedule).

**Option-Implied Beta forecaster** (Session 15 — built; the event-conditioned variant is now a blend arm in the default):
- `breeden_litzenberger.py` — PURE: extract the risk-neutral PDF `f_Q(K) = e^{rT}·∂²C/∂K²` from an
  option smile. IV-space smoothing cubic spline → BS reprice on a dense grid → numerical 2nd derivative
  → undiscount → normalise. `ImpliedPDF` (over terminal index prices) + `.to_return_distribution()`.
  `implied_pdf_from_iv` / `_from_calls` / `_from_chain` (duck-typed OptionContract).
- `beta.py` — `Cov/Var` rolling Beta vs SPY+IWM. `log_returns`, `align_log_returns` (inner-join),
  `compute_beta`, `BetaCalculator` → `BetaResult` (`.blended()`).
- `option_implied_beta_forecaster.py` — `OptionImpliedBetaForecaster`: blend SPY+IWM PDFs 50/50, draw a
  whole-horizon market return, scale by blended Beta, add a summed-daily idiosyncratic residual
  (`idiosyncratic_residuals` = `r_stock − β·r_market`). Systematic draw is one-per-path (PDF is for the
  expiry); idiosyncratic is `sum(horizon)` daily draws — so `forecast()`'s horizon must match the PDF's.

---

## backtest/ — evaluation

**`log_score.py`** —
- `rolling_log_score_backtest(ts, forecaster_factory, ...)` — per holdout day: build forecaster
  from current `ReturnDistribution`, forecast, score (PIT percentile + KDE log score), advance via
  `add_sample`. Leak-audited.
- `rolling_log_score_backtest_conditioned(ts, calendar, ...)` — event sibling: builds conditioned
  distributions from TRAINING data only (no price look-ahead) but uses KNOWN future event dates for
  the schedule. Per-iteration seed bump.
- `BacktestResult` — `mean_log_score`, `log_score_stderr`, `percentiles`, `to_dataframe`.

**`option_implied_beta_backtest.py`** (Session 16) — point-in-time rolling backtest of the
Option-Implied Beta forecaster. `build_eod_index_pdf(symbol, run_date, horizon, spot, ...)` builds the
as-of index PDF from historical EOD option bars (liquid monthly expiry, OTM put+call, volume-weighted
quadratic smile); `HistoricalIndexPdf` caches it per (symbol, run_date). `rolling_log_score_backtest_
option_implied_beta(ts, spy_ts, iwm_ts, index_pdf_fn, ...)` reuses `OptionImpliedBetaFactory` per
holdout day with point-in-time-truncated histories; skips (and counts) dates with no index PDF rather
than degrading to bootstrap. Driver: `scripts/run_option_implied_beta_backtest.py` (paired vs bootstrap).
Also `rolling_log_score_backtest_option_implied_beta_event` (event-conditioned idiosyncratic).

**`option_implied_backtest.py`** (Session 17) — rolling backtest of the own-options BENCHMARK.
`HistoricalTickerPdf` (cached, reuses the symbol-generic `build_eod_index_pdf`) +
`rolling_log_score_backtest_option_implied`. Drivers: `scripts/run_option_implied_backtest.py` (paired
vs event-bootstrap), `scripts/analyze_option_implied_quantiles.py` (per-eval edge by realized-return quantile).

**`blend_backtest.py`** (Session 18) — `rolling_log_score_backtest_blend_event(ts, spy, iwm, index_pdf_fn,
calendar, …) -> {eboot, oib, blend}`: builds the event-bootstrap and event-OIB forecasters in LOCKSTEP
per holdout day (components match their standalone backtests) and scores their 50/50 mixture. Driver:
`scripts/run_blend_backtest.py` (`--mode vs-eventboot` = compare + blend-vs-eventboot; `--mode vs-options`
= blend-vs-own-options).

**`standardized_test.py`** — `run_standardized_test(tickers, forecasters, ..., event_source=None)`.
`FORECASTER_CLASSES` registry (uniform forecasters); `event_bootstrap` handled via a separate path
(needs an `EventCalendar`). `StandardizedTestResult`: `summary_table`, `overall`,
`paired_comparison(a,b)`, `to_report`. Decorrelated per-iteration seeds + common random numbers
across forecasters.

---

## universe/ — standardized test set

**`snapshot.py`** — fetches S&P 500 (Wikipedia, browser UA) + caps (yfinance `fast_info`), FREEZES
to a dated CSV in `universe/snapshots/` (committed = source of truth; refresh deliberate).
**`sampler.py`** — `sample_tickers(n, seed, power=1.0, exclude=())` cap-weighted, without
replacement, seeded; independent of global RNG.

---

## CLI / scripts
- `scripts/run_standardized_test.py` — builds test set (META + 19 cap-weighted @ seed 42), runs
  forecasters, writes `output/standardized_test_<date>_h<H>_hold<HO>.{md,csv}`. Flags: `--events`,
  `--horizon`, `--holdout`, `--n-paths`, `--refresh-universe`, `--event-window`, `--anchor-experiment`
  (hold Bootstrap fixed, compare spot anchors: intraday_random vs stale_close).

## File map (key modules)
```
data/history.py, data/stock_return_ts.py, data/options_chain.py, data/spot_anchor.py, data/occ.py  # occ.py: OCC option-symbol parser
data/events/{event,source,yfinance_source,fomc_source,manual_calendar,composite,calendar}.py
data/options_history.py  # historical EOD option bars + list_contracts (expired discovery, status=inactive); provider Protocol, Alpaca→Databento
backtest/{option_implied_beta_backtest,option_implied_backtest,blend_backtest}.py  # as-of PDFs + rolling backtests (OIB / own-options / 50-50 blend)
forecast/{return_distribution,price_distribution,base,conditioned_returns,factories}.py  # factories.py: live forecaster wiring (BlendFactory = default)
forecast/{bootstrap,block_bootstrap,gaussian,event_bootstrap,garch_fhs}_forecaster.py
forecast/{breeden_litzenberger,beta,option_implied_beta_forecaster}.py  # option-implied beta (event variant = a blend arm in the default)
forecast/{option_implied_forecaster,blend_forecaster}.py  # own-options benchmark (S17) + N-way mixture (S18, default = eventboot⊕event-OIB)
valuation/{black_scholes,option_valuer}.py
sizing/kelly.py
execution/broker.py     # BUY-to-open (execute/submit_buy) + SELL-to-close (submit_sell); paper-only, dry-run default
portfolio/constructor.py  # greedy gross/directional/per-name allocation across candidates
run_daily.py            # ENTRY orchestrator + CLI (plan-all → allocate → execute-accepted); default forecaster = 50/50 blend (S18)
manage_positions.py     # EXIT orchestrator + CLI (reprice each open option → SELL-to-close when fair_value ≥ sell_threshold below bid)
backtest/{log_score,standardized_test,anchor}.py  # anchor.py: injectable spot-source + experiment
universe/{snapshot,sampler}.py
scripts/{run_standardized_test,run_option_implied_beta_backtest,run_option_implied_backtest,run_blend_backtest,analyze_option_implied_quantiles}.py ; config/manual_events.csv
```
