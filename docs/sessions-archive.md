# Session Log — Archive (Sessions 1–9)

Older session detail, split out of `docs/sessions.md` in S18's tech-debt cleanup to keep
that file's read cost low (per CLAUDE.md's start-of-session contract). Same chronological
narrative convention: current state lives in `docs/status.md`; decisions in
`docs/decisions.md`; numbers in `docs/results.md`.

---

## Session 9 (cont. 4) — 2026-06-05 — FOMC-inclusive event conditioning backtested (ACCEPTED)
- Ran the full 20-ticker `--events` standardized test with `FomcCalendarSource` in the event composite
  (earnings + FOMC + manual). **event_bootstrap −4.050 vs bootstrap −4.089 → +0.0388, t=2.54, p=0.011,
  55% win (11/20)** → clears the decision criterion. Marginally stronger than the prior earnings-only
  event result (+0.032, p=0.027, 50% win), i.e. adding Fed days didn't degrade and plausibly helped.
  Biggest single-name wins: FTNT (−3.79→−3.29), WMT, UNP, LLY, TMO, GOOGL.
- **Caveat recorded:** not a clean FOMC ablation — the run changes the event SET (and gives FOMC fuller
  history + BMO/announcement-day timing), so it doesn't isolate FOMC's marginal contribution. Added a
  todo: earnings-only vs earnings+FOMC head-to-head + BMO-vs-AMC timing test.
- Updated `docs/results.md` (leaderboard → current event source; prior earnings-only result preserved),
  CLAUDE.md decision-criterion record, status.md, todo.md. No code change. Artifacts:
  `output/standardized_test_2026-06-05_h5_hold63.{md,csv}`.

## Session 9 (cont. 3) — 2026-06-05 — FOMC / Fed meetings auto-fetched
- **Problem found:** macro events were FOMC-only AND the manual CSV was stale (last entry 2026-05-06,
  all in the past) → **no Fed meeting was reaching any live forecast** (forward schedule had 0 macro days).
- **`data/events/fomc_source.py` — `FomcCalendarSource`.** Scrapes the Fed's official FOMC calendar
  page (`federalreserve.gov/monetarypolicy/fomccalendars.htm`), which carries ~6 years (recent history
  + scheduled future) — so one fetch feeds both the conditioning distribution and the forward schedule.
  Pure `_parse_fomc_calendar(html)`: year-panel split, takes the LAST day of each meeting (the decision
  day), handles cross-month meetings ("Apr/May"+"30-1"→May 1), strips SEP `*` markers, excludes
  `notation vote` rows. Network isolated in `_get()` (browser UA + timeout, like snapshot.py);
  failure/empty → `EventSourceError` (composite skips it, run continues). Timing default BMO =
  announcement day's own close-to-close return (statement ~2pm ET); documented + parameterised.
- **Wiring.** `run_daily._default_forecaster_factory` composite is now earnings → FOMC → manual (order
  matters: the fetcher's auto-updating FOMC wins the (ticker,date,type) dedup over any stale manual row).
  Same source added to `scripts/run_standardized_test.py --events`. Manual CSV left as-is (harmless;
  fetcher wins overlaps) and is now for ad-hoc/non-FOMC macro.
- **Tests:** +10 (`tests/test_fomc_source.py`, no-network HTML fixture) → **297 pass**.
- **Live-verified:** parser pulls 43 dates 2022–2027 (all decision-day Wednesdays + the Nov-2024
  election-shifted Thursday); next FOMC 2026-06-17. NVDA (no earnings in window) now carries that Fed
  day on horizon day 7, built from 27 historical Fed-day samples → forecast std +3.1%.
- **Caveat / next:** macro conditioning is wired and internally consistent but the +0.032 event
  acceptance was earnings-driven — run `--events` standardized test to confirm FOMC conditioning lifts
  the log score (and to settle the BMO-vs-AMC timing) before treating it as a trusted edge.

## Session 9 (cont. 2) — 2026-06-05 — Event-conditioned forecaster wired into the LIVE path
- **Closed the central validate→trade gap.** The backtest-winning `event_bootstrap` and event
  awareness were offline-only (`run_daily` hardcoded plain `BootstrapForecaster`, never built a
  calendar). Now the live forecast is event-conditioned by default.
- **Forward event projection (the missing primitive).** `EventCalendar.forward_schedule(run_date,
  horizon_days)` — the old `horizon_event_schedule(future_dates=...)` was a `NotImplementedError`
  ("no run_daily orchestrator"). It projects the next K trading days (none exist in `ts` yet) and maps
  KNOWN upcoming events onto horizon days by laying them on a synthetic `[run_date, F0, F1, …]` axis
  and reusing `_return_index_for_event` — so the return-index convention makes the index == horizon
  day, and the BMO/AMC geometry is shared with the historical path. +7 tests.
- **Injectable forecaster factory.** `forecast/factories.py`: `ForecastContext`, `ForecasterFactory`,
  `bootstrap_factory`, `EventBootstrapFactory` (builds calendar from a source → conditions dists →
  forward schedule → event forecaster; degrades to bootstrap on any feed error), `event_days_in_horizon`.
  +5 tests (in-memory fake source, no network).
- **run_daily wiring.** `plan_ticker`/`run_daily` take `forecaster_factory`; `plan_ticker` default =
  `bootstrap_factory` (keeps unit tests no-network), `run_daily` default = `_default_forecaster_factory`
  (event-conditioned: yfinance earnings + `config/manual_events.csv`). New `TickerPlan.events_in_horizon`
  surfaced as an `ev` column in the summary. CLI: `--no-events`, `--event-window`. 4 run_daily tests
  now pass `forecaster_factory=bootstrap_factory` to stay offline.
- **Tests: 287 pass** (+12). **Live-verified:** ORCL(06-10)/ADBE(06-11)/FDX(06-23) show ev=1 in a 21d
  horizon, NVDA/WMT/COST ev=0; ADBE 14d dry-run forecast std widens +24.5% (22.4→27.9), downside 5%
  212.5→197.0 — the bimodal earnings effect the plain bootstrap was blind to. Docs updated
  (architecture-overview, architecture, runbook, todo, status).

## Session 9 (cont.) — 2026-06-05 — Architecture review + docs
- **Code review across all layers** (data → forecast → valuation → sizing → portfolio → execution).
  Key findings: (1) production `plan_ticker` hardcodes `BootstrapForecaster` and never builds an
  `EventCalendar` → the backtest-winning `event_bootstrap` and event awareness are NOT live (the
  central validate→trade gap); (2) no idempotency guard — a 2nd same-day `--live` double-submits;
  (3) no run persistence / fill reconciliation; entry-only (no exit); (4) per-ticker planning is
  sequential with ~4 network calls each (history + latest price + 2 chain pulls — `_select_expiry`
  and `plan_ticker` both fetch the chain); (5) config drift (`DEFAULT_N_PATHS` 50k vs 10k;
  `max_fraction` 0.20 vs 0.05); (6) `edge_pct_buy` == `expected_return_buy`. Full prioritized
  backlog (P0/P1/P2) in `docs/architecture-overview.md`.
- **New docs:** `docs/architecture-overview.md` (two-loops diagram, layer table, invariants,
  improvement backlog) and `docs/runbook.md` (weekly Monday run procedure + the backtest-gated
  change flow). Fixed a stale line in `docs/architecture.md` ("valuation, not built" → full
  pipeline). Registered both in CLAUDE.md docs index. No code changed; 275 tests still green.

## Session 9 — 2026-06-05 — Backtested the live-price anchor (ACCEPTED)
- **Q1 (where the latest-price anchor lives).** Two steps, two places: the forecaster only *applies*
  the anchor (`BootstrapForecaster.forecast(horizon, spot)` → `spot·exp(R)`); the *choice* of which
  price (live latest trade vs last completed close) is in `run_daily.plan_ticker` (orchestration —
  input pre-processing, not post-processing). Kept the forecaster pure (design invariant); exposed the
  anchor as an injectable source instead of moving I/O into it.
- **Built the anchor capability.** `backtest/anchor.py`: `AnchorContext` (frozen) + `stale_close_anchor`
  (close[t]) + `intraday_uniform_anchor` (uniform draw from next day's [low,high], synthesising an
  intraday price — no intraday ticks in the daily backtest) + `ANCHOR_FNS` registry. Added an optional
  `anchor_fn`/`anchor_seed` to `rolling_log_score_backtest` (default None → historical behaviour byte-
  for-byte; requires horizon≥2 when set). `run_anchor_experiment` packs the two anchors as the two
  "forecaster" keys of a `StandardizedTestResult` so the existing pooled paired t-test applies unchanged;
  CRN across arms (fresh per-arm decorrelated factory, shared base_seed). `--anchor-experiment` CLI flag.
- **Timeline modelled:** running intraday on the day AFTER close[t]. Stale anchor = close[t] (grid-aligned
  but a day old); intraday anchor = next day's [low,high] draw (fresh, off-grid by a partial day). Both
  forecast the same horizon to the same realised close[t+horizon]. The log score weighs staleness vs the
  off-grid slop — the production tradeoff.
- **Q2 result (full 20-ticker standardized set, h=5, holdout=63, n_paths=10k):** intraday_random −4.0138
  vs stale_close −4.0889 → **+0.0751, t=4.50, p=7.6e-06, 18/20 win → ACCEPTED.** Broad, larger than the
  event-conditioning lift. **Backtest-validates the S7 live-price anchor** (shipped on intuition; now proven).
  Caveat: the lift partly reflects the fresh anchor being ~1 day nearer the target than the close grid —
  which is exactly the real effect of anchoring on the live price. (4-ticker smoke first: +0.074, p=0.046.)
- **Production symmetry refactor.** Extracted `run_daily`'s inline live-vs-close block into
  `data/spot_anchor.py`: `AnchoredSpot(spot, source)` + `last_close_anchor` + `live_else_close_anchor`
  (default; calls `get_latest_price`, falls back to close) + a `SpotAnchor` type. `plan_ticker`/`run_daily`
  take an injectable `spot_anchor` param; `TickerPlan.spot_source` now records "live"/"last_close". This
  is the production mirror of `backtest/anchor.py` (intraday_random↔live_else_close, stale_close↔last_close),
  documented both ways. `run_daily` no longer imports `get_latest_price` (anchor module owns it) — test
  patches moved to `spot_anchor.get_latest_price`.
- **Tests:** +9 (`tests/test_anchor.py`) and +1 (`test_custom_spot_anchor_is_injectable`), anchor tests
  now also assert `spot_source` → **275 pass**, no-network. Docs: `results.md` head-to-head, CLAUDE.md
  decision-criterion record, todo.md item closed, architecture.md file map.
- **Next:** the fat-tail forecaster remains the top open forecaster experiment.

## Session 8 — 2026-06-05 — Second live multi-name run (no code changes)
- **Live run executed (`--live`).** Default 15-ticker watchlist, bankroll ~$94,656. Pre-run state:
  5 option positions held (AVGO C425 ×3, GOOGL C380 ×6, META P570 ×2, NEM C116 ×4, VRT C335 ×3,
  ~$10.7k premium); **no resting orders** (S7's NEM order had filled @2.90).
- Book 100% bullish → unhedged → budget cut 20% to $15,145 (16%); after charging the ~$10.7k open
  premium, only ~$4.4k free. Deployed **$4,140 (4.4%)** across 2 names:
  **CCI 6× C100 @0.43, GOOGL 6× C385 @6.47** → both **SUBMITTED, resting NEW** (limit-DAY at ask;
  thin/illiquid, hadn't filled within ~minute of placing; market open 09:43 ET). GOOGL C385 is a
  *distinct* contract from the held C380 — no double-submit.
- Rejected for `gross_budget`: ADI, AVGO, GE, GOOGL(C370 dup-strike candidate), LLY, MNST, NEM, VRT.
  5 no-BUY, 1 error (BRK-B — same `BRK.B` symbol bug, still unfixed). ORLY produced a BUY this run
  (no longer empty-chain).
- **Note:** dry-run preview (LLY ×2 + MNST ×56) differed from the live run (CCI + GOOGL) because the
  forecast re-anchors on live prices each call — intraday ticks reshuffled the greedy allocation.
- **Caveat unchanged:** still no idempotency guard. The two CCI/GOOGL-C385 orders may still be resting —
  check `get_orders` before any re-run today. No tests run / no code changed.

## Session 7 — 2026-06-04 — Live-price anchor + unhedged haircut + first live multi-name run
- **Unhedged haircut (replaces the S6 hard direction cap).** Dropped `max_direction_share`; added
  `hedge_threshold` (0.80) + `unhedged_haircut` (0.20) to `PortfolioCaps`. Constructor is now two-pass:
  allocate at full budget → if one side > 80% of deployed premium, cut gross budget 20% and re-allocate.
  `AllocationResult.hedge_haircut_applied` flag; `run_daily` summary reports it. Tests: replaced the
  direction-cap test with 3 haircut tests. See `docs/decisions.md`.
- **Live run executed (`--live`).** Full 15-ticker watchlist, bankroll ~$98,990. Book was 100% bullish
  → unhedged → budget cut 20% to $15,838 (16%); deployed $15,140 (15.3%) across 4 names. Orders:
  AVGO 3× C425 @15.96, GOOGL 6× C380 @7.77, VRT 3× C335 @15.10 → **FILLED**; NEM 4× C116 @2.90 → resting
  (new). Verified via `get_orders`/positions. (META 2× P570 from S6 still held.) Rejected for
  `gross_budget`: ADI, GE, LLY, MNST. 5 no-BUY, 2 errors (BRK-B, ORLY).
- **Two known data errors (unfixed):** BRK-B → Alpaca wants `BRK.B` (dot, not dash) — fails both history
  and chain; ORLY → empty chain for the selected expiry window. Isolated per-ticker; run completed.
- **Caveat for next live run:** no idempotency guard — re-running `--live` double-submits. Fill/position
  reconciliation still on the backlog.
- **265 tests pass** (was 263; +2 net from the haircut tests, after +2 live-anchor tests earlier).

### Session 7 (earlier) — Live-price forecast anchor
- Bug: `plan_ticker` anchored the forecast on `ts.close[-1]` (last *completed* daily close —
  intraday that's yesterday's print). Live demo: META forecast anchored on 622.98 (06-03 close)
  while the tape was 639.16 → forecast `S_T = spot·exp(R)` ~2.6% low, biasing every valuation.
- Fix: added `get_latest_price(ticker)` to `data/history.py` (Alpaca latest-trade, thread-bounded,
  returns None on any failure — creds/network/no-data). `plan_ticker` now anchors on the live price
  and falls back to last close (with a warning) when unavailable. `_select_expiry`/valuer/sizer all
  flow off the new spot.
- Impact on the META dry-run: spot 622.98→638.84; pick shifted put 605→put 590; edge 7.52→1.92;
  sizing 5 ct/$3,865 → 2 ct/$954. Confirms how much the stale anchor was inflating the bet.
- Tests: +2 (`test_spot_anchored_on_live_price`, `..._falls_back_to_close...`); fixture patches
  `get_latest_price`→None to stay no-network. **263 pass.**

## Session 6 — 2026-06-03 — Portfolio cap (cross-ticker allocation)
- Built `portfolio/constructor.py`: `PortfolioConstructor` / `PortfolioCaps` / `AllocationResult`.
  Greedy-by-expected-log-growth allocation under three caps: **gross premium** (≤20% bankroll = total
  capital-at-risk, since long-option premium = max loss); **directional balance** (≤60% of gross budget
  per side, calls=bullish/puts=bearish); **per-name** (≤5%). Existing open-option premium charged to
  budget (re-run safety). User chose: greedy, 20% gross, and — instead of estimating a correlation
  matrix — **directional balance** to bound market beta directly. (Decisions in `docs/decisions.md`.)
- Wired into `run_daily`: now plan-all → allocate → execute only accepted; rejected candidates keep
  `order=None` + an `alloc_reason`. +15 tests (constructor + run_daily allocation paths) → **261 pass**.
- 15-ticker live dry-run: the book that was 40.4% nearly-all-bullish (uncapped) → **12.7% deployed**,
  bullish capped at $10,985 (≤$12k), 6 calls rejected `direction_cap`, rest of budget idle (only META
  produced a bearish signal — the lopsided-signal case, handled as intended: cap the majority, never
  fabricate the minority). System is now ENTRY-ONLY and feature-complete for opening positions.
- Open before a real unattended run: fill reconciliation, idempotency guard, exit logic, market-hours
  awareness, cheap-option contract cap, edge quality (ISSUE-1). See `docs/status.md` + `docs/todo.md`.

## Session 5 — 2026-06-02 — Valuation layer + options-chain layer
- Built the `valuation/` layer (the differentiator): `black_scholes.py` (BS price + greeks + Brent
  implied-vol solver, European, kept as reference only) and `option_valuer.py`
  (`OptionValuer`, `OptionContract`, `OptionValuation`, `OptionType`, `Recommendation`). Consumes a
  `PriceDistribution` → discounted P-measure EV, prob-ITM, breakeven, and market edge (BUY/SELL/HOLD
  vs ask/bid). Cornerstone test: discounted EV over a risk-neutral lognormal `PriceDistribution`
  matches `bs_price` (ties the empirical machinery to the closed form).
- Built `data/options_chain.py`: `get_option_chain(underlying, *, option_type, expiry/strike windows,
  feed, include_untradable)` merges two Alpaca calls — `TradingClient.get_option_contracts` (tradable
  master list, paginated) + `OptionHistoricalDataClient.get_option_chain` (live bid/ask + IV) — into
  the valuation `OptionContract` objects. Mirrors `history.py` (CredentialsError propagates, worker-
  thread timeout, typed `OptionsDataUnavailableError`). Tradable-only by default; quotes are
  best-effort (snapshot failure → quote-less contracts, valuer reports NO_QUOTE). Pure `_build_contracts`
  merge is the unit-testable core. Feed defaults to "indicative" (free/paper; OPRA needs entitlement).
- +49 tests across the two layers → **210 pass**. All mocked → still needs ONE live paper smoke test
  of `get_option_chain` to confirm field mapping + feed entitlement against the real API.
- Decisions (see `docs/decisions.md`): fair_value = P-measure EV (not Q/no-arb price); held-to-expiry
  V1; European; per-share (multiplier deferred to Kelly); edge vs transacted side.
- Ran the first live forecast-vs-market comparison (META 2026-06-18): bootstrap forecast mean sat
  ~$3.45 below the risk-neutral forward → valuer flagged SELL all calls / BUY all puts. Logged as
  **ISSUE-1** (long-trend drift / no true-valuation anchor) with the full mean-reset/shrinkage
  discussion; **decided to defer** and stay on the execution path. New `docs/issues.md`.
- Built `sizing/kelly.py`: `optimal_kelly_fraction` (full-Kelly f*, validated against closed-form
  binary Kelly f*=p−q/b) + `KellySizer` (¼-Kelly default, hard max-fraction cap = the ISSUE-1 guard,
  100x multiplier, integer contracts, long-only). `KellySizing` result with status
  SIZED/NO_EDGE/NO_QUOTE/BELOW_ONE_CONTRACT. +18 tests → **228 pass**. Verified the full
  forecast→value→size chain on live META.
- Caveat surfaced: the per-position cap does NOT bound correlated aggregate exposure (9 puts ≈ one
  bet, ~22% total). Needs a portfolio-level cap — logged in `docs/status.md` next steps + ISSUE-1.
- Built `execution/broker.py`: `AlpacaBroker` — PAPER-only TradingClient (live intentionally
  unreachable), LIMIT orders only (default limit = contract ask, marketable; option spreads too wide
  for market orders), `dry_run` mode, pre-submit account guards (active / not blocked / enough
  options buying power) that SKIP rather than raise, injectable client for no-network tests.
  `execute(sizing) -> OrderResult` {SUBMITTED, DRY_RUN, SKIPPED, REJECTED}. +13 tests → **241 pass**.
- Live-verified against the paper account: read-only snapshot (ACTIVE, $100k, options level 3) +
  full-pipeline dry-run on META. Result correctly SKIPPED (BELOW_ONE_CONTRACT): ¼-Kelly budgets
  ~$1,375 but one ATM put costs ~$1,556 — sizer refuses to over-bet, broker honors it. No real order
  placed (left to explicit user confirmation).
- **Placed the first REAL paper order** (market closed → ACCEPTED, resting for next open, 0 filled):
  BUY 2× META260618P00570000 (META $570 put) @ $6.12 limit, ~$1,224 (¼-Kelly, 5% cap, f*=0.054).
  Order id aacedc2b-9423-4d4a-9161-9dbdc31e9c32. Proves the SUBMITTED path live (was mock-only).
  Caveat: signal is the deferred ISSUE-1 directional drift — execution validation, not conviction.
  DAY order placed while closed; multi-session resting would need GTC.
- Added **edge decomposition** diagnostic: `OptionValuer.decompose_edge` + `EdgeDecomposition` +
  `forward_price`. Re-centers our distribution onto the forward (pure multiplicative rescale,
  preserves shape) and re-values → splits each signal's edge as `total = drift + shape`. +5 tests →
  **246 pass**. For the purchased META 570 put: total +$2.09 = drift +$0.90 (43%) + shape +$1.18 (57%).
  Pattern across strikes: drift is the directional tilt (helps puts, hurts calls, monotonic in strike);
  shape (non-drift residual, includes the half-spread) is positive for most strikes → mild long-vol
  lean. `drift_share` >100% / negative when drift & shape have opposite signs (calls; deep-ITM puts) —
  expected, only OTM puts (40–75%) are cleanly interpretable. Directly operationalizes ISSUE-1.
- Built `run_daily.py` orchestrator (`run_daily` + `plan_ticker` + `_select_expiry` + CLI, dry-run
  default, one position/ticker, per-ticker error isolation). +6 tests → **252 pass**.
- **15-ticker live dry-run** (META + 14 large-caps): 0 errors, 11 sized BUY signals, 4 no-BUY, 40.4%
  aggregate deployment (uncapped across tickers). Findings: (1) plumbing scales cleanly; (2) ISSUE-1
  drift is UNIVERSE-WIDE — drift% mostly high, several >100% (shape negative) → system bets each
  ticker continues its recent trend (META down→put, most others up→calls); (3) no cross-ticker cap;
  (4) cheap OTM options size to huge contract counts (MNST 70, GOOGL 21). Engineering done; edge
  quality (ISSUE-1 + fat tails) and portfolio cap are the real open work.
- Next: cross-ticker portfolio cap; cheap-option contract limit; fill reconciliation; edge-quality work.

## Session 4 — 2026-06-02 — Notes reorg
- Restructured project docs for sustainable, low-cost session starts: added auto-loaded `CLAUDE.md`
  + split monolithic `NOTES.md` into `docs/{status,todo,architecture,decisions,results,sessions}.md`.
  `NOTES.md` is now a redirect. Rationale: Claude Code auto-loads `CLAUDE.md` (and walks parent
  dirs — the parent quant-research `CLAUDE.md` was polluting context; CLAUDE.md now disclaims it).
  @imports load eagerly, so heavy docs are referenced by plain path and read on demand.

## Session 3 — 2026-06-02 — Event conditioning
- Plan: `~/.claude/plans/yes-let-s-make-a-giggly-globe.md`. Built the event layer (`data/events/`),
  `ConditionedReturnDistributions`, `EventConditionedBootstrapForecaster`, wired into the backtest
  (`rolling_log_score_backtest_conditioned` + `event_bootstrap` in the standardized test, `--events`).
- Architecture: Option A (standalone event layer; `ReturnDistribution` stays pure). See
  `docs/architecture.md` (event layer + return-index mapping) and `docs/decisions.md`.
- Result: event_bootstrap best overall, +0.032 vs bootstrap, p=0.027 (accepted). 161 tests pass.
  Verified empirically there's no price look-ahead (uses known event dates only). See `docs/results.md`.

## Session 2 — 2026-06-02 — Block bootstrap + standardized test
- Built `BlockBootstrapForecaster` (moving-block, reconstructed from the daily `ReturnDistribution`)
  and the reproducible cap-weighted standardized test (`universe/` snapshot+sampler,
  `standardized_test.py`, `scripts/run_standardized_test.py`). New deps: requests, lxml, tabulate.
- Results: block bootstrap REJECTED at h=5 (p=2e-4) and h=10 (p=2.8e-7); kept for event use.
  Bootstrap ≈ Gaussian → bottleneck is fat tails. See `docs/results.md`, `docs/decisions.md`.
- Fixed report-filename clobbering (stamp `_h<H>_hold<HO>`).

## Session 1 — 2026-05-19 — Bootstrap forecaster end-to-end
- Built data layer (`history.py` → `StockReturnTS`), forecast layer (`ReturnDistribution`,
  `PriceDistribution`, `Forecaster` ABC, `BootstrapForecaster`, `GaussianForecaster`), backtest
  (`rolling_log_score_backtest` with PIT + KDE log score, leak-audited), and viz charts. 93 tests.
- Result: bootstrap ≈ Gaussian (indistinguishable, p=0.74); both show a PIT U-shape (fat-tail miss).
  Strategic call: default to MC/bootstrap (not parametric) because event-conditioning is on the
  roadmap and bootstrap extends to it naturally. See `docs/results.md`.
- Project facts: `pip install -e .`, module path `options_trader.*`, venv at `./venv`, Alpaca creds
  via env vars, no git remote.

