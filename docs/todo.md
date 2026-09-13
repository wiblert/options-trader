# To-Do Board (by service)

Canonical live backlog, organized by forecaster-pipeline service. Every forecaster item is
gated by the standing **log-score** decision criterion (see CLAUDE.md) before it becomes a default.

## ⭐ Architecture / tech-debt cleanup (added S18, resolved S19 — none changed behaviour)
Consolidated from the scattered "P2 cleanup" notes in `docs/architecture-overview.md` + debt found in
the S17/S18 builds.
- [x] **Centralize run config.** `config.py` already held the canonical live defaults; closed the
  remaining drift — `kelly.py`'s own stale `DEFAULT_MAX_FRACTION=0.20` now imports `config`'s 0.05, and
  5 scripts' hard-coded `risk_free_rate=0.04` now reference `DEFAULT_RISK_FREE_RATE`.
- [x] **Fix the `index`-named-but-symbol-generic helpers.** `build_eod_index_pdf` → `build_eod_pdf`,
  `HistoricalIndexPdf` → `HistoricalEodPdf`, `live_index_pdf` → `live_eod_pdf` (full rename, no aliases —
  internal API, no external consumers). Updated all callers/tests/docs.
- [x] **Blend default fetches SPY/IWM chains per ticker.** Added `LiveEodPdfCache` keyed on
  (symbol, run_date, horizon), shared across the whole watchlist run via
  `_default_forecaster_factory`'s `OptionImpliedBetaFactory(index_pdf_fn=...)`. **Caught by the
  end-to-end integration run (S19):** the first version cached with check-then-fetch-then-store,
  which the NEW thread-pooled planning (below) defeats — N ticker threads all miss the cold cache
  before any of them finishes the fetch, so SPY/IWM still fetched once per ticker (verified live: 4
  tickers → 4x SPY + 4x IWM chain fetches). Fixed to single-flight PER-KEY locking (hold the lock
  across the whole miss; other threads requesting the same key block and then reuse the result).
  Re-verified live: 4 tickers → exactly 1x SPY + 1x IWM. Regression test spawns 8 threads behind a
  start barrier and asserts exactly 1 underlying fetch (fails 8-vs-1 against the old logic).
- [x] **Two architecture docs overlap.** Decided the split explicitly in both headers:
  `architecture-overview.md` owns the ONE Layers table + two-loops diagram + invariants;
  `architecture.md` goes one level deeper per directory for the complex layers and does not repeat
  the table.
- [x] **Factory-mutation hardening** — `rolling_log_score_backtest` now passes `copy.deepcopy(rd)` to
  the factory.
- [x] **Dedupe the double chain fetch** — `plan_ticker` fetches the chain once (spanning both the
  expiry-discovery window and the valuation strike band) and `_select_expiry` takes that chain instead
  of re-fetching.
- [x] **Parallelize per-ticker planning** — `run_daily` plans tickers via a `ThreadPoolExecutor`
  (`max_workers`, default 8; `--workers` CLI flag; `1` = sequential).
- [x] **Per-ticker forecaster seed** — added `_ticker_seed(base_seed, ticker)` (CRC32-derived, deterministic);
  `plan_ticker` and `review_position` now seed per-ticker instead of sharing one seed across the watchlist.
- [x] **Collapse `OptionValuation.edge_pct_buy` / `expected_return_buy`** — removed the algebraically
  identical `expected_return_buy` field; `edge_pct_buy` is the one used downstream (review app).
- [x] **Trim the long-doc footprint** — split Sessions 1–9 out of `docs/sessions.md` into
  `docs/sessions-archive.md` (783 → 521 lines). `results.md` (419 lines) left as-is — a leaderboard doc,
  not narrative, not yet unwieldy.

## Data — `data/history.py`, `StockReturnTS`
- [x] **Live-price forecast anchor** (Session 7) — `get_latest_price()` (Alpaca latest-trade) so
  `plan_ticker` anchors the forecast on the current price, not the last completed daily close;
  falls back to last close when unavailable.
- [x] **Backtest the live-price anchor** (Session 9) — built `backtest/anchor.py` (injectable
  spot-source, forecaster stays pure) + `anchor_fn` hook in `rolling_log_score_backtest` +
  `run_anchor_experiment` (`--anchor-experiment`). Models running intraday on the day after a
  completed close: `stale_close` (close[t]) vs `intraday_random` (uniform in next day's [low,high]),
  CRN-paired, both scored to close[t+horizon]. **ACCEPTED** on the full 20-ticker set: intraday
  +0.0751 log-score, t=4.50, p=7.6e-06, 18/20 win — backtest-validates the live-price anchor.
  See `docs/results.md`.

## Events — `data/events/`
- [x] **FOMC / Fed meetings (auto-fetched)** (Session 9) — `data/events/fomc_source.py`
  (`FomcCalendarSource`) scrapes the Fed's official calendar (~6y history + scheduled future), wired
  into the live `run_daily` default and the `--events` backtest source. Replaces the stale hand-entered
  FOMC rows. Live-verified: NVDA (no earnings in window) now prices the 2026-06-17 FOMC, +3.1% forecast
  std. **Backtest-validated:** full 20-ticker `--events` run → event_bootstrap +0.039 vs bootstrap,
  p=0.011, 55% win (≥ the earnings-only +0.032/p=0.027), clears the decision criterion. Timing default
  = announcement-day return (BMO mapping).
- [ ] **Clean FOMC ablation + timing test** — the accepted run changes the whole event set, so FOMC's
  *marginal* lift isn't isolated. Run earnings-only vs earnings+FOMC head-to-head, and BMO-vs-AMC
  FOMC timing, on the standardized set; record in `docs/results.md`.
- [ ] **Import other external event sources / types.** Hypothesize catalysts that may carry a
  distinct return distribution, then source + backtest each: ex-dividend dates (yfinance
  calendar; small predictable gaps), macro prints beyond FOMC (CPI, NFP jobs, GDP — market-wide
  vol days), index add/drop (S&P inclusion — large idiosyncratic moves), triple-witching /
  monthly OPEX (elevated vol), analyst rating & price-target changes (single-name jumps), FDA
  decisions (bimodal binary outcomes), guidance / investor-day / product launches, peer-earnings
  read-through (e.g. NVDA → AVGO), IPO lock-up expirations. Backend: `AlpacaNewsSource`
  (Protocol impl) for unstructured headline/macro events; forward market calendar
  (`pandas_market_calendars`) for run_daily.
- [ ] **Smooth the event conditional return distribution.** Per-type distribution is built from
  ~11 raw earnings samples → bumpy/spiky. Apply KDE/jitter (`ReturnDistribution.smooth_samples`,
  Silverman default) to the event-keyed distributions; **pick the bandwidth by log-score lift,
  not eyeball.** Caveat: smoothing breaks sample time-ordering — fine for iid event bootstrap,
  NOT for a future block variant.
- [ ] **Cross-ticker pooling into the conditional distribution.** Per-ticker earnings history
  (~11) is thin; pool other tickers' event-day returns (via the universe sampler). Likely
  vol-standardize per ticker before pooling; keep the target's own samples (optionally
  upweighted). Backtest whether the extra data beats the loss of idiosyncrasy.
- [ ] **Split event bootstrapping into OVERNIGHT vs TRADING-day (intraday) components.** Earnings
  (AMC/BMO) and many catalysts move the price as an OVERNIGHT GAP (prior close → next open), with a
  different distribution from the subsequent intraday (open → close) move. Today we condition on the
  single close-to-close event-day return, which blends the two. Decompose each event day into
  overnight (gap) + intraday return, build/condition them separately, and recombine in the horizon
  simulation (gap on the event day, normal intraday otherwise). This needs OHLC (we have open/close
  on `StockReturnTS`) — overnight = log(open[t]/close[t-1]), intraday = log(close[t]/open[t]).
  Hypothesis: the gap carries most of the event's fat tail/skew while intraday is closer to normal, so
  separating them sharpens the event distribution. **Gate on log score.** Ties into the BMO/AMC timing
  question (which return the event attaches to) and the event-OIB idiosyncratic conditioning.

## Visualization / Diagnostics — `webapp/`, `viz/`
- [x] **Forecast diagnostic web UI** (Session 11) — `webapp/forecast_service.py` (pure payload builder)
  + `webapp/server.py` (stdlib http.server, Plotly CDN) + `static/index.html`. Random/typed ticker →
  spaghetti (30 paths) / terminal PDF / CDF, normal (lognormal) overlay, median highlighted; diagnostic
  mode holds out `horizon` bars and reports realized PIT percentile + KDE log score. Launch:
  `python scripts/run_webapp.py`.
- [ ] **Event-conditioned overlay in the webapp** — spaghetti + terminal dist currently use the plain
  iid bootstrap (illustrative). Add the production `EventBootstrapFactory` forecaster as a third
  distribution/path set so the UI shows what actually trades (toggle; degrade to bootstrap on feed failure).
- [x] **Options & recommended-price panel** (Session 14) — superseded by the interactive **review app**
  (`webapp/review_service.py` + `static/review.html`, route `/review`): per held position and per sized
  BUY candidate it shows the forecast distribution with the strike/spot/E[S_T]/breakeven overlays, the
  `OptionValuer` fair value / edge / recommendation, and Kelly sizing — and lets you approve to execute.
- [x] **Human-in-the-loop daily run** (Session 14) — review app over `run_daily`: guided sell-then-buy
  queue, Kelly budget meter, approve = execute immediately (dry-run toggle). Backend reuses
  `review_position`/`plan_ticker`/`PortfolioConstructor` with no execution; `forecast_sink` hook on the
  orchestrators feeds the charts the actual production distribution.
- [ ] **⚠️ Review app won't render in the browser (Session 14, OPEN — do first)** — `/review` shows a
  blank page / spinner over the VS Code SSH port-forward, though the server + API are proven via curl
  (200, ~1 ms) and Plotly is now vendored locally (no CDN). Diagnose browser-side: DevTools Console/
  Network on the blank page, confirm the forwarded local port, `curl http://localhost:<port>/review`
  from the laptop, try incognito. See `docs/status.md` next-step item 0 + `docs/sessions.md` S14.
- [ ] **Review app follow-ups** (after it renders) — fill/status polling after an approve (currently
  fire-and-forget like the CLI), and a market-hours / live-creds preflight before enabling live paper.

## Return Distributions — `forecast/return_distribution.py`, `conditioned_returns.py`
- (nothing outstanding)

## Forecasters — `forecast/*_forecaster.py`
- [x] **Wire the event-conditioned forecaster into the LIVE path** (Session 9) — `forecast/factories.py`
  (`ForecastContext` + injectable `ForecasterFactory` + `EventBootstrapFactory`) + `EventCalendar.forward_schedule`
  (forward event projection; the old `future_dates` NotImplementedError). `run_daily` default is now
  event-conditioned (yfinance + manual macro CSV), `--no-events` to opt out. Live-verified: ADBE 14d
  forecast std +24.5% with the earnings day routed. Closes the validate→trade gap on the forecaster seam.
- [ ] **Fundamentals-blended forecaster** (addresses ISSUE-1 in `docs/issues.md`) — the current
  forecast is pure historical-return extrapolation with NO estimate of the ticker's true/fair value,
  so it expresses only a noisy directional drift (live META test → SELL all calls / BUY all puts).
  Add a fundamentals-based fair-value anchor the distribution drifts toward (mean-reversion to a
  target), blended with the historical forecast. **Measure via the rolling log-score backtest before
  adopting** — not theory. Consider also a de-meaned-onto-forward variant to isolate pure vol/shape edge.
- [~] **Option-Implied Beta forecaster** (S15 built, **S16 backtested — promising, NOT accepted**).
  Components: `data/options_history.py` (EOD bars + `list_contracts` expired discovery, provider
  Protocol Alpaca→Databento), `forecast/breeden_litzenberger.py` (`f_Q(K)=e^{rT}·C''(K)`, default
  volume-weighted quadratic-in-log-moneyness smile), `forecast/beta.py` (rolling Cov/Var β vs SPY+IWM),
  `forecast/option_implied_beta_forecaster.py` + `OptionImpliedBetaFactory`,
  `backtest/option_implied_beta_backtest.py` + `scripts/run_option_implied_beta_backtest.py`.
  **Backtest (S16, 100 tkr, h=21, hold=63, eval≈Feb–Apr 2025):** oib−bootstrap **+0.046 (p=1.6e-12,
  77/100)**, oib−garch_fhs **+0.103 (p=3e-23, 58/100)** — clears the criterion vs BOTH incumbents on
  this window. Convex per-ticker pattern (big wins on high-vol/beta names, small losses on low-vol
  defensives). **⚠️ BUT sanity check failed:** garch_fhs−bootstrap = −0.058 (vs known +0.137) → the
  window flatters forward-looking vol; edge magnitude likely regime-inflated; default NOT flipped.
  **FULLY BACKTESTED (S16) — plain + event, 2 windows. Borderline, default NOT flipped:** vs bootstrap
  it's a vol-stress regime artifact (W1 +0.046 / W2 ≈0); vs the event-GARCH-FHS default event-OIB wins
  both windows (+0.048 / +0.026) but softly (GARCH weak in these 2024–25 windows). **Next (pick one):**
  (a) tie-breaker window where event-GARCH recovers ≈+0.224; (b) **vol-stress regime gate** for OIB
  (its edge is large+robust only when vol is elevated — most evidence supports this); (c) OIB+event-GARCH
  **ensemble** (different edges). Else shelve as built-and-measured. Deferred per user: P-measure
  de-meaning, VRP variance. Also open: `live_eod_pdf` is calls-only (EOD builder uses OTM puts+calls —
  unify); BL strike-range tail truncation. See `docs/results.md` (event 2-window table).
  ✅ S15 gating blocker (historical index PDF per past date) RESOLVED — `get_option_contracts(status=
  INACTIVE)` + `get_option_bars(Day)`; see `docs/sessions.md` S16.
- [~] **Ensemble / router across forecasters** (motivated by the S16 per-ticker analysis) — OIB and
  event-bootstrap (and GARCH-FHS) draw on DIFFERENT information sources, so they win on different names
  and should combine well. **S18 — approach (d) path-level 50/50 mixture BUILT + backtested**
  (`BlendForecaster`/`BlendFactory`/`blend_backtest.py`/`run_blend_backtest.py`): blend(event-bootstrap,
  event-OIB) beats event-bootstrap **+0.0107, p≈6e-14, 27/40 names** (1yr, 40 tkr) but does NOT beat OIB
  (−0.002, n.s.) — gain ~85% inherited from OIB; blend is a robustness play (between both arms on 33/40,
  lower per-ticker variance). vs own-options: +2.16 pooled but thin-chain-inflated (liquid-only +0.73,
  median −0.026). **Default NOT flipped** (single window; OIB regime-dependent). Next gate: held-out 2nd
  window + the **vol/beta router** (a/below). See `docs/results.md` S18. Evidence (`docs/results.md`
  "Where each forecaster dominates"): per-ticker
  best over 2 windows splits **OIB 24 / event-GARCH 21 / event-bootstrap 14** of 59; the split is
  largely by **volatility/beta** and STABLE across regimes — OIB wins high-vol/high-beta names
  (UAL/RL/META/NVDA/semis/cyclicals), event-bootstrap wins low-vol defensives (utilities/REITs/insurers/
  BRK-B), corr(OIB−boot edge, vol level)=+0.55. The pooled W2 "tie" was the AVERAGE of OIB's high-vol
  wins and defensive losses cancelling — i.e. a real cross-sectional signal hidden by pooling.
  **Approaches to backtest (gate each on log score, on a held-out window to avoid overfitting the
  combiner):** (a) **vol/beta router** — hard-switch to OIB for high-vol/high-beta names, event-bootstrap
  for low-vol defensives (threshold fit on training only); (b) **per-ticker historical-best selector**
  (pick the arm with the best trailing OOS log score per name); (c) **log-score-weighted blend / stacking**
  (softmax of recent per-arm log scores → weights); (d) **path-level mixture** (draw a fraction of MC
  paths from each forecaster's terminal distribution — cleanest probabilistically, preserves each shape).
  Start simple (a or d). Watch: the router must be point-in-time (no look-ahead in the switch signal);
  judge on pooled log score AND per-regime robustness (both windows), not one favorable window.
- [ ] **Fat-tail forecaster** (Student-t jitter or `JumpDiffusionForecaster`) — addresses the
  Session-1 PIT U-shape; the remaining gap (block AND Gaussian both ≈ bootstrap). Highest-value
  forecaster experiment. Slots into `FORECASTER_CLASSES` + standardized test.
- [x] **Filtered Historical Simulation (FHS)** (Session 11) — `forecast/garch_fhs_forecaster.py`
  (GARCH(1,1) variance-targeting QMLE in scipy → filter → bootstrap standardized residuals → propagate
  σ² forward). **ACCEPTED:** 40-tkr h=21 **+0.137, p=0.0002, 57% win**, best of all forecasters. Convex
  improvement (big wins on vol-regime names, slight drag on calm names). See `docs/results.md`.
- [REJECTED — S11] **FHS vol-divergence GATE.** `VolGatedForecaster` (FHS when σ_{T+1}/σ̄ ≥ τ, else
  bootstrap), τ-swept on 40 names h=21: monotonically harmful — plain always-on FHS (τ=0) is the best
  arm; every gate worse, several worse than pure bootstrap. FHS wins across the whole vol profile (incl.
  calm days, by narrowing correctly), so any threshold discards wins. Ship plain FHS ungated. A per-NAME
  gate (persistence/ARCH-LM) is a different untested idea — not pursued. See `docs/results.md`/`decisions.md`.
- [x] **Wire the winning FHS variant into the LIVE path** (Session 12/13) — `GarchFhsForecaster`
  (unified; event params keyword-only optional): normal days GARCH-filtered, event days draw from the
  empirical event distribution (earnings/FOMC), GARCH state advances from the shock. **ACCEPTED** h=21
  20-tkr +0.224 vs bootstrap, p=0.000276, 65% win. `GarchFhsFactory` is the production default in
  `_default_forecaster_factory`. `garch_fhs_event_forecaster.py` deleted (merged into `garch_fhs_forecaster.py`).
  See `docs/results.md`.
- [ ] **Option-implied variance as a vol-level input** — feed implied variance minus a calibrated
  *constant log* VRP per name (Carr-Wu: single-name VRP is unreliable at the level, well-behaved in
  logs). Gate on log score. See research note.
- Note (Session 11 research): the CENTER/drift question is effectively settled — trailing drift is
  near-unestimable and our null dampening result matches the literature (Merton/Goyal-Welch/Hjalmarsson).
  EV is in TAILS/SHAPE + VOL LEVEL, not the center. Full cited write-up:
  `docs/research-forecast-accuracy-2026-06.md`.
- [ ] **Event-conditioned BLOCK bootstrap** — block lost unconditionally but should capture
  post-catalyst multi-day path shape; route block windows by event type.

## Backtest / Evaluation — `backtest/`
- [ ] **Factory-mutation hardening** — `rolling_log_score_backtest` passes `rd` straight to the
  user factory; a factory that mutates it (e.g. `smooth_samples()`) corrupts later iterations.
  Pass `copy.deepcopy(rd)`.
- Note: per-iteration seed decorrelation is DONE in `standardized_test` and the conditioned
  backtest; the plain `rolling_log_score_backtest` still uses a fixed factory seed.

## Universe — `universe/`
- [ ] **Edge-aware watchlist sampling (replace cap-weighting).** Today `run_daily` samples the watchlist
  cap-weighted (∝ √cap, `power=0.5`). But cap is a liquidity/safety proxy, not where our FORECAST has
  EDGE. Two hypotheses to backtest (gate on realized log-score edge vs the market price, not intuition):
  (a) **success-weighted** — weight each ticker by our historical out-of-sample edge over the options
  market (per-name mean `forecast − option_implied` log score, e.g. from `run_blend_backtest.py
  --mode vs-options` / the S17 own-options study), so we trade names where we've actually beaten the
  market PDF. (b) **inverse-liquidity / low-volume tilt** — sample UNWEIGHTED by cap, or up-weight
  LOWER-volume names, on the hypothesis that we do relatively better on less-liquid names (their option
  PDF is noisier / market is less efficient — the S17 own-options gaps were largest on thin chains, and
  the OIB edge concentrated in higher-vol names). Watch-outs: (1) selection/survivorship + look-ahead —
  the success weights must come from a PRIOR window, not the eval window (else we curve-fit the
  watchlist); (2) lower-volume names have worse FILLS/slippage and thinner chains (more skips), so a
  "forecast edge" there may not be a TRADEABLE edge — judge on net-of-cost P&L, not just log score;
  (3) keep a liquidity floor so we don't sample untradeable names. Implement as a new
  `sample_tickers` weighting mode + a backtest comparing watchlists. Ties to S17 own-options quantile
  finding + the OIB vol/beta router.
- [x] **Dedup dual-class issuers (GOOGL/GOOG) — S20.** Two distinct problems, confirmed by reading
  `universe/snapshot.py`/`sampler.py`/`data/history.py`/`data/options_chain.py`: (A) GOOGL/GOOG,
  FOX/FOXA, NWS/NWSA are separate snapshot ROWS for the same issuer — `sample_tickers` could draw both
  into one watchlist, silently doubling that company's effective exposure past the per-name cap;
  (B) `BRK-B` is a SEPARATE pure string-format bug (one row, one company — no `BRK-A` row exists to
  dedup against), unrelated to A. Fixed A: `universe/sampler.py::DUAL_CLASS_ISSUERS` (3 known groups)
  + `_dedup_dual_class` keeps the higher-market-cap ticker per group (self-adjusting if cap ordering
  flips, not a hardcoded "primary" side — confirmed against the real snapshot: FOXA and NWS are each
  the kept ticker despite my first draft assuming the other side). Wired into both `sample_tickers`
  and the `sampling_weights` audit helper.
- [x] **Symbol-format mapping for Alpaca — S20.** Problem B above: `BRK-B` (yfinance/snapshot dash
  format) is rejected by Alpaca (wants `BRK.B`). Added `data/symbol_format.py::to_alpaca_symbol`
  (small static dash→dot table: `BRK-B`, `BF-B`) applied ONLY at the Alpaca request boundary in
  `history.py` (`_from_alpaca`, `get_latest_price`) and `options_chain.py` (`_fetch_contracts`,
  `_fetch_snapshots`) — the yfinance fallback still gets the untranslated dash ticker, and
  `OptionContract.underlying`/`StockReturnTS.ticker` are stamped with the CALLER's canonical ticker,
  not Alpaca's own dot-format response, so downstream grouping (portfolio caps, held-type guards)
  stays keyed consistently. Live-verified: `get_history("BRK-B")` and `get_option_chain("BRK-B")` both
  succeed against real Alpaca data (previously errored on every live run since Session 7); full
  `run_daily --tickers BRK-B,AAPL` dry-run plans BRK-B with 0 errors.
- [x] **ORLY/empty-chain handling — root-caused + fixed (S10 cont.5).** Was a DTE-window miss, not a
  thin chain: `_select_expiry`'s `[target−10, target+14]` window could fall between monthly expirations,
  so monthly-only names (most mid-caps) returned "No tradable contracts." Widened the upper bound to
  `target+35` (always spans a monthly cycle). 40-name dry-run errors 18→3. Residual genuinely-unlisted/
  symbol-format names remain: **BRK-B** (dash-vs-dot, see universe todo), **PCAR**, **UDR**.

## Valuation — `valuation/`
- [x] `black_scholes.py` — analytical BS price + greeks + implied-vol solver (reference only).
- [x] `option_valuer.py` — `PriceDistribution` + contract → discounted P-measure EV, prob-ITM,
  breakeven, market edge (BUY/SELL/HOLD). Held-to-expiry V1. (Session 5)
- [ ] Early-exit valuation (deferred from V1) — needs the price distribution + IV at an interim
  exit date; only if held-to-expiry proves too limiting.

## Options data (downstream — needed for execution) — `data/`
- [x] `options_chain.py` — `get_option_chain(underlying, ...)` merges Alpaca `get_option_contracts`
  (tradable master list + OI, paginated) + `get_option_chain` (live bid/ask/IV) into `OptionContract`
  objects. Mirrors `history.py` creds/timeout/typed errors; tradable-only; quotes best-effort. (Session 5)
- [x] **Live paper smoke test** (Session 5) — pulled a real META chain: 812 contracts, all quoted,
  IV present. Confirmed the "indicative" feed returns full quotes + IV on the paper account (no OPRA
  entitlement needed) and field mapping (strike/expiry/type/bid/ask/IV/OI) is correct on live data.
  Greeks intentionally dropped — revisit only if a hedging/diagnostic need appears.

## Sizing / Execution (downstream) — `sizing/`, `execution/`
- [x] `sizing/kelly.py` — `optimal_kelly_fraction` (full-Kelly f*) + `KellySizer` (¼-Kelly, hard
  max-fraction cap, 100x multiplier, integer contracts, long-only V1). (Session 5)
- [ ] **Aggregate / correlated exposure cap** (portfolio layer) — the per-position Kelly cap does not
  bound a basket of correlated bets (live META: ~2.5% into each of 9 puts = one ~22% directional
  view). Add a total-exposure and/or correlation-aware cap. Ties to ISSUE-1.
- [ ] **Short-side sizing** — Kelly for written/short options (different loss domain, 1+fR can go
  ≤0). Deferred; V1 buys only.
- [x] `execution/broker.py` — `AlpacaBroker` paper-only limit-order placement + account/position
  snapshots, dry-run, account guards, injectable client. (Session 5; live read + dry-run verified)
- [x] `run_daily.py` — orchestrator (`run_daily` + `plan_ticker` + `_select_expiry` + CLI) wiring
  forecast → chain → value → size → `broker.execute` over a watchlist. Dry-run default, `--live` to
  submit, one position/ticker, per-ticker error isolation. (Session 5; 15-ticker dry-run verified)
- [x] **Cross-ticker portfolio cap** (`portfolio/constructor.py`) — greedy-by-log-growth under gross
  (20%) + per-name caps; charges existing premium; wired into run_daily. (Session 6.) **Session 7:**
  the S6 hard directional cap (≤60%/side) was REPLACED by an unhedged-book haircut — one side > 80% of
  deployed premium ⇒ cut the gross budget 20% and re-allocate (`hedge_threshold`/`unhedged_haircut`).
- [ ] **Joint / correlation-aware Kelly** (future upgrade to the directional-balance heuristic) —
  correlated joint simulation + multivariate `E[log(1+Σ fᵢRᵢ)]`. Judge on portfolio P&L/drawdown.
- [REJECTED — S10] **Cheap-option contract/notional limit.** Considered (Kelly buys large counts of
  cheap OTM options within the dollar cap) and rejected: tails are the model's edge (not its weak spot),
  concentration is already bounded by the per-name/gross caps + diversification, limit orders handle
  fills, and Kelly oversizing is already covered by fractional-Kelly + the max-fraction cap. The
  thesis-aligned move is a better tail model (fat-tail forecaster), not a cap. See `docs/decisions.md`.

## Live-run readiness — `execution/`, `run_daily.py`
- [x] **Single execution command (Session 13)** — `run_daily` now runs both passes: EXIT
  (`manage_positions` — reprice holds, sell-to-close overpriced) then ENTRY (plan watchlist, fund
  signals). `--sell-threshold`, `--no-manage` CLI flags. `DailyRunResult.manage_result` carries the
  exit output. Combined summary prints exit table first. `manage_positions.py` remains standalone.
  First live combined run: 1 SELL submitted (CCI260618C00100000 ×6 @ $0.03), 0 new entries
  (gross budget exhausted by existing positions).
- [ ] **Fill/position reconciliation** — poll order status after submit, confirm fills via
  `get_positions`, persist holdings. Decide DAY vs GTC.
- [~] **Idempotency / re-run guard** — open-orders awareness DONE (S10 cont.3): `broker.get_open_orders()`
  feeds both passes — a resting BUY counts toward the entry one-call/one-put-per-name guard, a resting SELL
  blocks exit resubmission. STILL OPEN: a deterministic day-scoped `client_order_id` as the broker-enforced
  backstop, and the submit→"appears open" race window.
- [x] **Exit logic — value-driven sell (Session 10)** — `manage_positions.py` reprices every open
  option (OCC parse via `data/occ.py` → reforecast to the option's expiry → value) and submits a
  sell-to-close limit at the bid when `(bid − fair_value)/bid ≥ sell_threshold` (default 0.15); else
  HOLD. Added `broker.submit_sell`. Live-verified: SOLD META260618P00570000 ×2 @ $5.23. **Still open:**
  expiry handling (currently a SKIP), partial-close sizing (sells the full position), and a time-stop /
  max-hold rule — exit is purely value-driven today.
- [ ] **Market-hours awareness** — broker doesn't check the clock; make resting-vs-fill explicit.
- [x] **First real paper order placed** (Session 5): BUY 2× META260618P00570000 @ $6.12, ACCEPTED
  (market closed, resting for next open). Proves the SUBMITTED path live.
- [ ] **Order management** — cancel/replace stale unfilled limit orders; handle partial fills; idempotent
  re-runs (client_order_id) so a re-run doesn't double-submit.
