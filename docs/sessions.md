# Session Log

Chronological narrative (history). Current state lives in `docs/status.md`; decisions in
`docs/decisions.md`; numbers in `docs/results.md`. Append a brief entry at each session end —
don't duplicate the detail docs, point to them.

---

## Session 18 — 2026-06-23 — 50/50 blend forecaster (event-bootstrap ⊕ event-OIB)

User asked to join the event-bootstrap and Option-Implied Beta forecasters with a simple 50/50 blend
(probabilities still sum to 100, no learned weights — "don't overuse the data") — the to-do
"Ensemble / router" approach (d), path-level mixture.

Built: `forecast/blend_forecaster.py` (`BlendForecaster` + pure `blend_price_distributions` — stack
both forecasters' weighted price samples and halve each side's weights so the mixture sums to 1,
preserving each shape — a true mixture, not an average of means); `factories.py::BlendFactory` (NOT a
default); `backtest/blend_backtest.py::rolling_log_score_backtest_blend_event` (eboot/oib/blend scored
in lockstep, components matching their standalone backtests); `scripts/run_blend_backtest.py` (`--mode
vs-eventboot` = items 1+3a, `--mode vs-options` = item 3b). +10 tests → 450 pass.

Results (1yr, h=21, holdout=252; OIB = event-conditioned): blend > event-bootstrap +0.0107 (p≈6e-14,
27/40) but NOT > OIB (−0.002, n.s.) — gain ~85% from OIB; blend is a robustness play (between both arms
on 33/40). Item 1: OIB beats event-bootstrap on high-vol/cyclical names (APH/CEG/IEX/AMD/AMZN), loses on
defensives (UNH/SBUX/CBOE). vs own-options (item 3b): +2.16 pooled but thin-chain-inflated; liquid-only
+0.73 with median −0.026 (win 47%) → tail-driven edge. Numbers: `docs/results.md`; ADR (50/50 vs
learned): `docs/decisions.md`.

**Then, per user direction, FLIPPED the production default to the blend.** `run_daily.
_default_forecaster_factory` → `BlendFactory([EventBootstrapFactory, OptionImpliedBetaFactory(
event_source=...)], 0.5/0.5)`. To make the LIVE OIB arm match the backtested event-OIB, added an optional
`event_source` to `OptionImpliedBetaFactory` (earnings-conditioned idiosyncratic residuals via
`_conditioned_residuals`, lazy-imported to avoid a cycle). `BlendForecaster` gained an `event_schedule`
passthrough so the run_daily `ev` column still reports. Regression test updated →
`test_default_forecaster_factory_is_blend`. Live-verified: `python -m options_trader.run_daily --tickers
AAPL --no-manage` dry-run forecasts AAPL (spot 294.28, call-310 signal, edge 1.47) with 0 errors. 450
tests pass. Caveat recorded: single-window edge; held-out 2nd window + vol/beta router still the proper gate.

---

## Session 17 — 2026-06-23 — Option-Implied (own-options) BENCHMARK forecaster

User asked for a way to compare our stock-price forecast against the PDF implied by the options
currently for sale — a backtest + forecast component to tell us whether our forecast is more
accurate than (and therefore can profit against) the market price — explicitly NOT for the
production engine, since it just regenerates the prices for sale.

Built (reusing existing machinery — `build_eod_index_pdf` / `live_index_pdf` are symbol-generic,
`ImpliedPDF.to_return_distribution`, the log-score backtest plumbing):
- `forecast/option_implied_forecaster.py::OptionImpliedForecaster` — reconstructs a ticker's OWN
  Breeden-Litzenberger risk-neutral PDF and re-anchors it to spot. Deterministic.
- `forecast/factories.py::OptionImpliedFactory` — live path, degrades to bootstrap. NOT wired as the
  default (`run_daily._default_forecaster_factory` unchanged = `EventBootstrapFactory`).
- `backtest/option_implied_backtest.py` — `rolling_log_score_backtest_option_implied` +
  `HistoricalTickerPdf` (cached, skips+counts dates with no PDF, never degrades).
- `scripts/run_option_implied_backtest.py` — paired head-to-head vs event-bootstrap; reports the sign
  of (option_implied − event_bootstrap): NEGATIVE ⇒ our forecast beats the market price (edge).

**Why benchmark-only:** valuing the same options it was built from is circular ⇒ ~zero edge by
construction. ADR in `docs/decisions.md`, ISSUE-N in `docs/issues.md`. Contrast: option-implied BETA
(index PDF + Beta + idio) is non-circular and still a genuine candidate.

+10 synthetic tests → **440 pass**. Live smoke vs Alpaca historical chains (META, 2024-06→2025-03,
h=21 holdout=30, n=10, 0 skipped): option_implied −5.688 vs event_bootstrap −5.453, diff −0.235
(t=−2.0, p=0.075) — illustrative only, confirms the pipeline. Numbers: `docs/results.md`; artifact
`output/option_implied_backtest_smoke.md`.

---

## Session 16 cont.4 — 2026-06-22 — GARCH complexity NOT justified over event-bootstrap

User asked for the bootstrap-vs-GARCH standardized check ("high bar for the complexity GARCH adds").
`scripts/run_garch_vs_bootstrap_check.py` on the canonical set (META + 19, seed 42), h=21, holdout=63,
recent window: garch_fhs_events −4.889, event_bootstrap −4.920, bootstrap −5.073. **event_bootstrap
+0.153 vs bootstrap (p=0.005); garch_fhs_events +0.184 vs bootstrap; head-to-head garch_fhs_events −
event_bootstrap = +0.031, t=0.87, p=0.38 → NOT significant.** So ~83% of GARCH-FHS's lift is the EVENT
conditioning, not the GARCH machinery — the GARCH(1,1) QMLE + FHS + vol-propagation adds an insignificant
+0.03 over the far simpler event-bootstrap (confirms the EWMA-already-tracks-recent-vol intuition).
**DONE: switched the production default `_default_forecaster_factory` GarchFhsFactory →
EventBootstrapFactory** (GARCH-FHS still available via `--garch-fhs-events`/`--garch-fhs`) — nearly all
the lift, no per-iteration GARCH fit, and makes the planned OIB + event-bootstrap ensemble a clean
two-source combination. Added a regression test pinning the default
(`test_default_forecaster_factory_is_event_bootstrap`) → 430 tests pass. Also added an **ensemble to-do**
(vol/beta router; OIB + event-bootstrap have complementary sources per the cont.3 per-ticker analysis)
and an **overnight/intraday event-split to-do**. Artifacts: `output/garch_vs_eventboot_2026-06-22_h21_hold63.md`.

## Session 16 cont.3 — 2026-06-22 — event-conditioned two-window three-way (borderline; not flipped)

Ran the fair fight the user asked for: event-bootstrap / **event-GARCH-FHS (real default)** / **event-OIB**
on both windows (60 tkr, h=21, hold=63). **vs event-bootstrap:** W1 +0.120 / W2 −0.001 (regime-dependent,
same as plain). **vs the event-GARCH default:** OIB wins BOTH windows (W1 +0.048 p=0.003, W2 +0.026 p=6e-4,
~55–58% of tickers) — the only cross-window-positive comparison. **Soft win caveat:** event-GARCH-FHS
underperformed its leaderboard +0.224 in both windows (W1 +0.072, W2 −0.027 vs event-bootstrap), so these
2024–25 windows are GARCH-unfavourable — "OIB > event-GARCH" is partly GARCH weakness, not OIB strength.
**Verdict: borderline-positive vs the real incumbent, not a clean robust win → default NOT flipped.**
Robust takeaways: OIB's big edge is vol-stress-conditional (regime gate is the live angle); it's never
much worse than baseline with big stress upside (ensemble/hedge member). Next options (status.md A):
tie-breaker window where GARCH is healthy, a vol-stress gate backtest, or an OIB+GARCH ensemble.
Artifacts: `output/oib_3way_event_w{1,2}.md`, `oib_3way_event_both.log`. 429 tests (no src change since cont.2).

## Session 16 cont.2 — 2026-06-22 — 2nd window: OIB edge is a REGIME ARTIFACT + event machinery built

Ran the 2nd non-overlapping window (200 tkrs, eval ≈ mid-2024, plain three-way) and built the
event-conditioned comparison the user asked for.

**2-window plain verdict:** W1 (Feb–Apr 2025 vol stress, 100 tkr) oib−bootstrap +0.046 (p≈1e-12) but
**W2 (calm mid-2024, 198 tkr) oib−bootstrap −0.005 (p=0.31, 85/198 ≈ coin flip), oib−garch −0.032.**
Crucially the garch−bootstrap **sanity check flips sign** — −0.058 in W1 (wrong), +0.026 in W2 (right,
≈its true +0.137 direction) — proving W1 is a regime where forward-looking option vol is flattered, not
a harness bug. **So plain OIB's edge is a vol-stress regime artifact, NOT a robust unconditional default.**

**Event machinery (user: "these should all be containing events"):** added event mode to
`OptionImpliedBetaForecaster` — idiosyncratic residuals routed to an earnings-conditioned distribution
(`_conditioned_residuals` partitions the residual series), while MARKET events (FOMC/CPI) stay free via
the index PDF (conditioning them on the idiosyncratic side would double-count). New
`rolling_log_score_backtest_option_implied_beta_event` + `--events` three-way in the runner
(event-bootstrap via `rolling_log_score_backtest_conditioned`, event-GARCH-FHS via
`rolling_log_score_backtest_garch_fhs_event`, event-OIB). Also engaged the user's EWMA insight: the
bootstrap is decay-weighted (λ=0.99, ~69-day half-life) so it already tracks recent vol crudely — which
is why plain GARCH's marginal edge over it is modest/fragile. +5 tests → 429 pass. Event three-way on
both windows (60 tkr) RUNNING; verdict pending. Artifacts: `output/oib_3way_200tkr_w2.{md,log}`.

## Session 16 cont. — 2026-06-22 — 100-ticker three-way (clears the bar on one window; robustness unconfirmed)

Scaled the gate to 100 cap-weighted-√ tickers (seed 42, power 0.5), h=21, holdout=63, window
2023-01→2025-05 (eval ≈ Feb–Apr 2025), n_paths=10k. Added a **plain GARCH-FHS** arm + universe sampling
+ artifact output to `scripts/run_option_implied_beta_backtest.py`. n=4300, 0 skipped, 0 failures.

**Result:** oib −4.184, bootstrap −4.230, garch_fhs −4.287. **oib−bootstrap +0.046 (t=7.1, p=1.6e-12,
77/100 tickers); oib−garch_fhs +0.103 (t=10.0, p=3e-23, 58/100).** Convex per-ticker: big wins on
high-vol/beta names (UAL +0.48, RL +0.30, ISRG +0.28, META +0.22, AAPL +0.19), small losses on low-vol
defensives (CVX/INVH/BRK-B ≈ −0.08).

**⚠️ Sanity check failed:** plain GARCH-FHS = −0.058 vs bootstrap, opposite its leaderboard +0.137 (S11).
Same harness, same scoring → the **eval window** is the cause: Feb–Apr 2025 (tariff-vol stress) rewards
forward-looking option-implied vol (OIB) and punishes GARCH's backward-looking conditional vol. So the
OIB edge is real on this window but its magnitude is regime-inflated and **not robust on one window**.
**Did NOT flip the production default.** Next: a 2nd non-overlapping window where garch−bootstrap recovers
≈+0.137 (confirm harness + isolate regime); OIB must keep its lift there. Per user, P-measure de-meaning
and VRP variance haircut are deferred. No src/ change → 424 tests. Artifacts: `output/oib_3way_100tkr.{md,log}`.

## Session 16 — 2026-06-22 — backtested Option-Implied Beta (got the backtest going)

Got the backtest going. **Confirmed via the live Alpaca paper API** that historical EOD option chains
are available for backtesting (the S15 "gating blocker"): expired contracts list with
`get_option_contracts(status=INACTIVE, expiration_date=<past>)` (active/none give nothing for past
expiries); daily bars via `get_option_bars(TimeFrame.Day)`; data since **Feb 2024**.

Built: `data/options_history.list_contracts` (expired discovery, paginated);
`backtest/option_implied_beta_backtest.py` (`build_eod_index_pdf` as-of index PDF from EOD bars,
`HistoricalIndexPdf` cache, point-in-time `rolling_log_score_backtest_option_implied_beta` that reuses
`OptionImpliedBetaFactory` and skips+counts dates with no PDF); `scripts/run_option_implied_beta_backtest.py`.

**Refactored `breeden_litzenberger`** — default smile fit is now a volume-weighted **quadratic in
log-moneyness** (`method="quadratic"`; spline kept as an option). Getting a clean PDF from EOD data was
the hard part (full write-up in the memory note [[alpaca-historical-options-api]]): use the liquid
MONTHLY expiry (weeklies have no historical bars), bars EXACTLY on run_date, OTM puts + OTM calls (deep-
ITM call EOD closes are stale → bias mean ~6% low), a volume filter, and the quadratic fit (a raw
spline's wiggles clip to negative C'' and halve the density width — annIV came out ~0.09 vs ATM IV 0.22).

**Result** (10 tkr, h=21, holdout=63, 2024-06→2025-05, 0 dates skipped, paired vs bootstrap):
option_implied_beta −5.104 vs bootstrap −5.183 → **+0.078, t=1.37, p=0.17, 7/10 tickers positive**;
lift CONVEX, concentrated in fat-tail names (WMT +0.219, META +0.199, JPM +0.185; per-date win 48%).
Directionally positive and thesis-consistent (Q-measure index tails help where big moves land) but NOT
significant ⇒ **does NOT clear the decision criterion; NOT a default.** Comparison was vs bootstrap;
the production incumbent is GARCH-FHS (+0.224), so OIB must clear a higher bar. +18 tests → 424 pass.
Numbers: `docs/results.md`. Next: 20-tkr + 2nd window (tighten p), head-to-head vs GARCH-FHS,
P-measure de-meaning of the systematic draw. Artifact: `output/oib_backtest_10tkr.log`.

## Session 15 — 2026-06-22 — Option-Implied Beta forecaster (built; NOT yet a default)

Built the four-component **Option-Implied Beta** framework: infer the market's risk-neutral PDF from
the most liquid index options (SPY large-cap, IWM small-cap), then map that systematic risk onto a
single ticker by its Beta and add back idiosyncratic risk. The thesis: individual mid-cap chains are
too illiquid for a clean direct PDF, but index options carry a smooth forward-looking distribution.

Components (see `docs/status.md` "Last worked on" for the per-file detail):
- `data/options_history.py` — historical EOD option bars for BACKTESTING. `OptionBarsProvider` Protocol
  (Alpaca→Databento swappable) + `AlpacaOptionBarsProvider` (`get_option_bars`, `TimeFrame.Day`). Pure
  `_bars_from_alpaca_df` core. Production bypasses it for live snapshots.
- `forecast/breeden_litzenberger.py` — `f_Q(K) = e^{rT}·C''(K)` via IV-space smoothing spline → BS
  reprice on a dense grid → numerical 2nd derivative → undiscount → normalise. `ImpliedPDF` +
  `to_return_distribution()`. Verified vs the analytic lognormal (flat-vol smile) and on skew recovery.
- `forecast/beta.py` — rolling `Cov/Var` Beta vs SPY+IWM, `align_log_returns`, `BetaCalculator`.
- `forecast/option_implied_beta_forecaster.py` — MC engine (50/50 PDF blend · Beta + summed daily
  idiosyncratic residual). `OptionImpliedBetaFactory` in `factories.py` (injectable index-PDF +
  history fns; live default = live chain→BL; degrades to bootstrap on any failure).

+40 synthetic tests → **415 pass**.

⚠️ **NOT wired as the production default.** Per the standing log-score criterion it must first beat the
incumbent (`GarchFhsFactory`) out-of-sample. The gating blocker is the historical index PDF per past run
date (needs as-of contract discovery — Alpaca's contract list is live-only — or Databento). Design
decisions to revisit at backtest time: risk-neutral (Q) vs realised (P) systematic draw (variance risk
premium), call-only smile (put-parity extension), strike-range tail truncation inherent to BL. Next
step + open sub-items: `docs/status.md` "Recommended next step A".

## Session 14 — 2026-06-17 — interactive review web app (human-in-the-loop run_daily)

Built a browser front-end so the daily pipeline's recommendations are reviewed and approved before
they trade, instead of `run_daily` firing orders unattended.

### What shipped
- **`webapp/review_service.py`** (new) — `build_review_session(broker, tickers, …)` runs the SAME
  pipeline as `run_daily` but stops short of any order: reuses `manage_positions.review_position`
  (EXIT, held-position sell reviews), `run_daily.plan_ticker` (ENTRY, sized BUY candidates) and
  `PortfolioConstructor.allocate` (recommended set + Kelly gross budget), then serialises everything to
  JSON. Each sell/buy carries a faithful **chart** payload: trailing close history + the *actual*
  production `PriceDistribution` terminal PDF (KDE on a price grid) + scalar overlays (spot, E[S_T],
  strike, breakeven, fair value). `execute_item(session, item_id, override)` runs one approved item via
  `broker.submit_sell` / `broker.execute` and tracks `deployed_so_far` vs the gross budget (over-budget
  buys are blocked unless overridden).
- **`run_daily.plan_ticker` + `manage_positions.review_position`** — added an optional `forecast_sink`
  callback (default `None`, no-op) invoked right after `forecaster.forecast(...)` so the review service
  captures the exact distribution used for the recommendation without re-deriving it. No behaviour change.
- **`webapp/server.py`** — added `do_POST`, an in-memory `SESSIONS` store, and routes `GET /review`,
  `POST /api/review/run` (owns network edges: `sample_tickers` + `AlpacaBroker`), `POST /api/review/execute`.
  Existing forecast routes untouched.
- **`static/review.html`** (new) — guided one-at-a-time queue (sells first, then buys), dark Plotly UI:
  two-panel decision view (history + forecast band; terminal distribution with strike/spot/breakeven/
  E[S_T] lines), metric cards, a Kelly **budget meter**, **Approve = execute immediately** (one-click
  confirm), and a **dry-run toggle** (default on). Manual ticker box + "use daily 40-name sample".
- **`tests/test_review_service.py`** (new) — +7 synthetic/no-network tests (stub broker +
  monkeypatched `get_history`/`get_option_chain`): session build, chart faithfulness, held-type
  exclusion, sell/buy execution dispatch, over-budget block→override. **375 pass** total.

### Verified (backend only)
Backend end-to-end vs the Alpaca **paper** account via **curl** (not the browser): `/api/review/run`
returned bankroll ~$82k, 12 held positions repriced as sell reviews, a `META260710C00595000` buy
candidate sized to 2 contracts; the over-budget guard fired (existing premium already > 20% gross cap)
and the override executed; the dry-run buy returned `DRY_RUN` and the budget meter decremented by $2,716.
No forecaster change → no log-score gate involved.

### Then: vendored Plotly locally
Downloaded `plotly-2.35.2.min.js` → `webapp/static/vendor/`; added a `/static/<path>` route to
`server.py` (`_send_static`, traversal-guarded, ext→MIME map); repointed both `index.html` and
`review.html` off `cdn.plot.ly` to `/static/vendor/...`. Bundle serves 200 (4.5 MB); traversal blocked.

### ⚠️ UNRESOLVED — UI never rendered in the browser (pick up here next week)
Viewing `/review` (and `/`) from the laptop over the VS Code SSH port-forward of :8000 gave a **blank
white page / infinite spinner**. Server is proven fine: `curl localhost:8000/review` on the VM → 200 in
~1 ms; the vendored Plotly → 200. Suspected the laptop's network was blocking the Plotly CDN (blocking
`<script>` in `<head>` → blank page), which is why we vendored it — but after server restart + browser
hard-refresh it STILL did not render. So either the CDN wasn't the (only) cause or there's a second
problem. **Not yet diagnosed.** Next session, work the browser side:
- Open DevTools **Console + Network** on the blank page; read the real error (JS exception? a request
  pending/failed? which URL?).
- In the VS Code **PORTS** tab confirm the *actual* forwarded local port (8000 may have remapped), then
  from the laptop: `curl http://localhost:<port>/review` — 200 means it's a render/JS issue, refused/hang
  means a forwarding issue.
- Try a trivially-different browser / incognito (rules out cache + extensions).
See `docs/status.md` "Recommended next step" item 0 for the same checklist.

---

## Session 13 — 2026-06-16 — merged manage_positions into run_daily + forecaster cleanup

### Part 1 — forecaster cleanup
- Merged `GarchFhsEventForecaster` into `GarchFhsForecaster`. Event params (`conditioned`,
  `event_schedule`) are now keyword-only optionals on the unified class. Plain mode (no event
  params) is bit-identical to the old plain `GarchFhsForecaster`.
- Deleted `src/options_trader/forecast/garch_fhs_event_forecaster.py`.
- Updated `factories.py`: `GarchFhsEventFactory` → `GarchFhsFactory` (now instantiates the
  unified constructor with event params); updated docstring.
- Updated `log_score.py`: stale comment in `rolling_log_score_backtest_garch_fhs_event`.
- Updated `tests/test_garch_fhs_event_forecaster.py`: rewritten for the unified constructor;
  added `test_normal_only_schedule_is_consistent_with_plain_garch_fhs` (bit-exact parity).
- 368 tests pass throughout.

### Part 2 — single execution command
- `run_daily` now runs the EXIT pass (`manage_positions`) first, then the ENTRY pass.
  Motivation: one `python -m options_trader.run_daily` replaces the two-step
  `run_daily` + `manage_positions` workflow.
- New `run_daily()` params: `sell_threshold: float = 0.15`, `skip_manage: bool = False`.
- New CLI flags on `run_daily`: `--sell-threshold`, `--no-manage`.
- `DailyRunResult` gains `manage_result: Optional["ManageRunResult"] = None`.
- Circular import avoided: `manage_positions` is imported lazily inside `run_daily()` function
  body (not at module level); `ManageRunResult` is imported under `TYPE_CHECKING` only.
- `_print_summary()` prints the exit table first (via a lazy import of `manage_positions._print_summary`)
  when `manage_result` is set, then the entry table.
- `manage_positions.py` unchanged — still runnable standalone.
- All 6 direct `run_daily(...)` calls in `tests/test_run_daily.py` now pass `skip_manage=True`
  to prevent the mocked broker from hitting the exit pass.
- 368 tests pass.
- `docs/runbook.md` updated: removed "ENTRY-ONLY today" caveat, merged Steps 1–3 to document
  the combined command, added exit-pass docs and `--no-manage` note.

---

## Session 11 (cont. 2) — 2026-06-12 — GARCH-FHS forecaster (ACCEPTED)
- **Goal:** build + backtest the research note's top pick — GARCH-filtered Filtered Historical Simulation.
- **Built:** `forecast/garch_fhs_forecaster.py` — `fit_garch11` (variance-targeting Gaussian QMLE over
  (α,β) via scipy SLSQP; lfilter recursion; fallback params; no new deps) + `GarchFhsForecaster` (filter
  → standardized residuals `z` → bootstrap `z` and propagate `σ²` forward, anchored on today's σ_{T+1}).
  Registered in `FORECASTER_CLASSES`; CLI `--garch-fhs`. +7 tests (fit stationarity, unit-variance z,
  conditional width responds to recent vol, reproducibility, validation) → 355 pass.
- **Backtest:** 20-tkr h=5 +0.027 (p=0.27, 45%); 20-tkr h=21 +0.107 (p=0.034, 55%); **40-tkr h=21
  +0.137, t=3.74, p=0.0002, 57% win** — best of all 4 forecasters, biggest lift of any idea to date.
- **Robustness read (40-tkr):** CONVEX improvement — median per-ticker +0.030; huge wins on vol-regime/
  fat-tail names (QCOM +3.69, FTNT +2.79, SBAC, AMZN, BX), slight drag on calm names (BSX/AMD/HUM/TXN/MU).
  Ex-FTNT still +0.069 (NOT one-name; QCOM > FTNT), but ex-top-3 → −0.047 and per-ticker t(n=40) p=0.28.
  Passes on the pooled per-forecast test (project standard; mean is the right objective for option-EV/
  Kelly, and a convex profile is desirable). Recorded in `results.md`/`decisions.md`/`issues.md`/CLAUDE.
- **Vol-divergence gate — built + REJECTED.** `VolGatedForecaster` (use FHS when σ_{T+1}/σ̄ ≥ τ, else
  bootstrap) + `run_vol_gate_experiment` τ-sweep + `--vol-gate-experiment` CLI. 40-tkr h=21: gating is
  monotonically harmful — τ=0 (always FHS) −5.037 is the best arm; every gate worse, several worse than
  pure bootstrap. Insight: the biggest loss is switching the CALMEST days off FHS, i.e. FHS wins on calm
  days too (correctly narrows when cond vol < baseline) — the edge is the whole vol profile, so any
  threshold discards wins. Calm-NAME drag is a per-name fit issue, not per-day. +5 tests → 360 pass.
- **Decision: ship plain GARCH-FHS ungated.** Only open step: wire it into `factories.py` as the LIVE
  default (separate deployment decision); event-conditioned FHS is a natural follow-up. Recorded in
  `results.md` / `decisions.md` / CLAUDE ledger.

## Session 11 (cont.) — 2026-06-12 — drift-dampening forecaster experiment (REJECTED)
- **Goal:** backtest a modification — bootstrap as usual, then shrink the terminal mean toward spot by a
  tunable `drift_dampening` κ (κ=1 = mean all the way to spot), and fit κ.
- **Built:** `forecast/drift_dampened_bootstrap_forecaster.py` (subclasses BootstrapForecaster; draws,
  then `price' = price + κ·(spot − mean)` — a pure additive translation, so dispersion/shape are
  preserved and only the centre moves). `backtest/standardized_test.py`: `run_drift_dampening_experiment`
  (one arm per κ, κ=0 arm is the CRN-paired incumbent ≡ plain bootstrap) + `best_dampening_arm`.
  `scripts/run_standardized_test.py`: `--drift-dampening-experiment` / `--kappas`. +6 tests → 348 pass.
- **Result (standardized 20-ticker set, n_paths=10k, holdout=63):** indistinguishable at every κ.
  h=5 best κ=0.25 = +0.0009 (t=0.57, p=0.57, 45% win); h=21 best κ=0.10 = +0.0008 (t=0.49, p=0.63,
  50% win). Full reset κ=1.0 strictly worse at both horizons. **REJECTED** by the decision criterion.
- **Finding:** per-ticker effect is large but ~symmetric and cancels — κ helps names whose recent drift
  was noise (META/TMO/WMT/WM/SBAC/HUM/WMB/PLTR), hurts names whose drift was signal (BSX/GOOGL/FTNT/
  KLAC/AAPL/NVDA/AVGO/UNP/LLY). A uniform global reset can't separate signal from noise → nets to zero.
  Empirically confirms the ISSUE-1 tension; the longer-horizon lift shrinkage theory predicted did not
  appear. Live path remains the *conditional* fundamentals anchor (ISSUE-1), not a global drift reset.
  Recorded in `docs/results.md`, `docs/decisions.md`, `docs/issues.md`.

## Session 11 — 2026-06-12 — forecast diagnostic web UI (`webapp/`)
- **Goal:** a webapp to visualise + diagnose forecast accuracy, using the existing pipeline functions.
- **What was built:** new `src/options_trader/webapp/` package.
  - `forecast_service.py` — PURE, network-free payload builder. Takes a `StockReturnTS` + horizon →
    JSON dict: trailing OHLC bars, 30 bootstrap **spaghetti** paths, p10/p50/p90 bands + **median path**,
    and terminal **PDF + CDF** for the empirical bootstrap (Gaussian KDE) AND a Gaussian/normal baseline
    (analytic lognormal from the SAME `ReturnDistribution`). **Diagnostic mode** holds out the last
    `horizon` bars → realized continuation + realized terminal's **PIT percentile** + **KDE log score**
    per model (the project's own accuracy metrics, so the page genuinely diagnoses quality).
  - `server.py` — stdlib `http.server` (ZERO new deps; Plotly via CDN). Routes `/`,
    `/api/random_ticker` (cap-weighted `sample_tickers(power=0.5)`), `/api/forecast` (wraps `get_history`
    + the service, translates failures to clean 400s). Binds 0.0.0.0:8000 by default.
  - `static/index.html` — single-page UI: ticker + 🎲 random, horizon, diagnostic/live mode, view toggle
    (spaghetti / terminal PDF / CDF), normal-overlay + band checkboxes, summary + accuracy cards, and a
    **reserved "Options & recommended price"** panel for the future.
  - `scripts/run_webapp.py` — launcher. +7 synthetic service tests → **342 pass**.
- **Verified:** server boots; page (14KB), random-ticker, and error paths return correctly; live
  AAPL diagnostic over Alpaca → spot $294.80, 30×22 spaghetti, boot/gauss log scores −4.076/−4.048
  (consistent with the backtest's ~−4.05).
- **Caveat / follow-ups:** spaghetti paths use the plain iid bootstrap mechanism (illustrative), NOT the
  event-conditioned production forecaster; terminal dists are the bootstrap-vs-normal head-to-head. Future:
  add the event-conditioned forecaster as a third overlay, and fill the options panel (chain + model vs
  market price + recommended price). See `docs/todo.md`.

## Session 10 — 2026-06-09 — verified future-event inclusion + live daily run (1 order)
- **Goal:** ensure the daily runner includes future events, then run it live for today.
- **Future events — verified, no code change.** Confirmed the Session-9 wiring works end-to-end:
  `EventBootstrapFactory` builds the calendar with a 400-day lookahead and calls
  `EventCalendar.forward_schedule(run_date, horizon)`, which projects upcoming events onto horizon days.
  Live check (NVDA, run_date 2026-06-09, 16-day horizon): the **2026-06-17 FOMC** lands on horizon day 5
  and is routed to `EventType.MACRO_FOMC` (`event_schedule[5] = MACRO_FOMC`). yfinance returns 13 earnings
  + FOMC source returns 33 Fed events in-window. In the run summary every priced ticker shows `ev≥1`
  (ADBE `ev=2`: FOMC + upcoming earnings). Sources confirmed to return future-dated events.
- **Pre-flight (idempotency, since there's still no guard):** account equity ~$89k; 7 open option positions
  from S5–S8 (~$9.4k MV) — the S5 META put DID fill (qty=2). **No resting/open orders** before the run
  (prior CCI/GOOGL/NEM/VRT/AVGO all filled), so no double-submit risk.
- **Live run** (`run_daily --target-dte 21 --live`, default 15-ticker watchlist, seed 42): 7 sized signals,
  100%-bullish book → 20% unhedged haircut → 16% gross budget (~$14.25k), minus ~$9.4k existing premium →
  only ~$4.8k free. **1 order funded + SUBMITTED: BUY 3× AVGO260702C00405000 @ limit $14.74 ($4,422).**
  Verified at broker: status `new` (resting DAY limit at ask, filled_qty=0). All other signals rejected
  `gross_budget`; 5 chain errors (BRK-B symbol format + CCI/MNST/OMC/ORLY thin/no chain — known backlog).
- **Note:** contract drifted dry-run→live (AVGO 410→405, NEM dropped) because live quotes re-priced between
  the two runs; expected with the live-price anchor. No code changed; no tests run (read/run-only session).
- **Open:** AVGO order is resting — verify fill before any further `--live` today (no idempotency guard).

### Session 10 (cont.) — EXIT pass: reprice holdings → sell-to-close (`manage_positions.py`)
- **Goal:** handle existing positions — reprice them and sell the ones the market now overpays for.
- **`data/occ.py`** — pure OCC option-symbol parser (`AVGO260702C00405000` → underlying/expiry/type/strike
  via the fixed 15-char tail). Unit-tested incl. fractional strikes, short roots, malformed inputs.
- **`broker.submit_sell`** — sell-to-close limit order (`OrderSide.SELL`, DAY, dry-run default, no
  buying-power guard since selling frees capital; Alpaca rejects over-sells → REJECTED). Broker is now
  BUY-to-open / SELL-to-close; still never opens shorts.
- **`manage_positions.py`** — EXIT orchestrator mirroring run_daily's safety (dry-run default, per-position
  error isolation, held-to-expiry forecast at the option's expiry, event-conditioned forecaster). Rule:
  reprice each open option, SELL-to-close at the bid when `(bid − fair_value)/bid ≥ sell_threshold`
  (default 0.15), else HOLD. Pure `decide_action` is separately tested. CLI: `--sell-threshold`,
  `--no-events`, `--live`.
- **+21 tests → 318 pass.** Dry-run repriced all 8 open positions; the 7 calls HOLD (fair value above
  bid — drift-heavy forecast, ISSUE-1), the lone put flagged SELL.
- **Live run** (user chose the 15% threshold): **SUBMITTED 1 sell — META260618P00570000 ×2 @ limit $5.23**
  (our fair $4.21, ~20% below bid). Verified at broker: status `new`, resting. Earlier AVGO 405 BUY
  filled (no longer in open orders).
- **Open / caveats:** no idempotency (a re-run would resubmit a sell for a name with a resting sell —
  checked open orders manually first); expiry handling is just a SKIP; sells the full position (no
  partial); exit is purely value-driven (no time-stop). Logged in status.md recommended-next-step.

### Session 10 (cont. 2) — per-ticker guard → one call + one put (holdings-aware)
- **Change:** `run_daily`'s concentration guard (ISSUE-1) was `max_positions_per_ticker=1` (keep the
  single best contract per name). Reshaped to: keep the **best CALL and best PUT** per ticker (each by
  expected log growth), and **exclude any type already held** on that name, so {held + ordered} ≤ one
  call and one put per ticker. Allows a genuine two-sided/vol view while still blocking correlated
  stacking (e.g. 9 puts on one name).
- **Impl:** `plan_ticker` now takes `held_types` and selects best-of-each-type instead of a top-N slice;
  `run_daily` reads open option positions once up front (parsed with `data/occ.py`) to compute both the
  budget premium charge and the per-name held types. Removed the `--max-per-ticker` CLI flag and the
  `max_positions_per_ticker` param. `_current_option_premium` → `_open_option_positions` +
  `_held_types_by_underlying`.
- **Tests:** replaced the old max-per-ticker tests with one-call-one-put, best-of-type, held-type
  exclusion (unit) + an end-to-end run_daily case where an open META put blocks new META puts. **320 pass.**
- **Live-verified (dry-run):** with our current book (calls on AVGO/CCI/GOOGL/NEM/VRT), those names now
  return "no BUY" (their call signals are excluded), while NVDA (unheld) still funds its best call.
- **Note:** the guard keys on *positions*, not resting orders — a resting (unfilled) BUY of a type isn't
  counted yet. Counting open orders belongs to the deferred idempotency guard. (Done next, cont. 3.)

### Session 10 (cont. 3) — open-orders awareness (partial idempotency guard)
- **Goal:** fold resting orders into the guards so a same-day re-run doesn't double-submit.
- **`broker.get_open_orders()`** — new typed read (`OrderSnapshot`: symbol/side/qty/limit/status/id) over
  Alpaca `GetOrdersRequest(status=OPEN)`.
- **Entry (`run_daily`):** `_held_types_by_underlying` → `_committed_types_by_underlying(positions,
  open_orders)`. A resting BUY now counts as a committed long toward the one-call/one-put-per-name guard;
  a resting SELL does NOT (it's a pending close, already represented by the position it closes). So
  re-running entry won't add a 2nd call/put while the first is still resting.
- **Exit (`manage_positions`):** before submitting, read open orders; if a SELL is already resting on a
  symbol, skip resubmission (review stays action=SELL, order=None, reason notes "resting sell exists").
- **+6 tests → 325 pass:** `get_open_orders` field mapping; resting-BUY blocks same-type entry (end-to-end);
  `_committed_types_by_underlying` counts positions+buys but not sells; exit skips/here-submits with/without
  a resting sell.
- **Live-verified:** `get_open_orders` reads cleanly in both passes. The earlier META put sell-to-close
  FILLED in the interim (now 7 option positions, 0 resting orders), so the live skip couldn't be exercised
  this run — covered by unit tests.
- **Still open (the rest of idempotency):** a deterministic day-scoped `client_order_id` as the
  broker-enforced backstop, and the submit→"appears open" race window. Logged in status.md step 2.

### Session 10 (cont. 4) — watchlist sampling: rotate daily, flatter weighting, 40 names
- **Why:** the default sampled watchlist was the same ~15 mega-cap names every run — seed hardcoded to
  42 AND pure cap-weighting (`power=1.0`, top-10 names = 43% of sampling probability).
- **Changes (CLI/`main` only):** `DEFAULT_N_TICKERS=40`, `DEFAULT_WATCHLIST_POWER=0.5` (∝ √cap).
  Seeds split: `--seed` = forecaster Monte-Carlo paths (still 42); new `--watchlist-seed` defaults to
  `_daily_watchlist_seed(date.today())` = run-date ordinal → rotates daily, stable within a day. New
  `--power` flag. `_default_watchlist(n, seed, power)` passes `power` to `sample_tickers`. main() prints
  the seed+power used for reproducibility.
- **Verified:** today's 40-name draw mixes mega-caps (AAPL/NVDA/GOOGL/META/JPM/COST) with many mid-caps
  (AEE/AJG/BLDR/CARR/CRH/EBAY/EXPD/HBAN/HIG/INVH/UDR/WAT/WRB) that never appeared under seed-42/power-1.0;
  consecutive days give disjoint draws. 325 tests pass (no test touched this CLI path).
- **Caveat:** 40 names ≈ 2.7× per-run API load (history + chain each) → slower, more thin-chain/symbol
  errors (per-ticker isolated, so non-fatal). `BRK-B` symbol-format issue still unfixed in the universe.

### Session 10 (cont. 5) — widened `_select_expiry` window (monthly-only names)
- **Bug surfaced by the flatter watchlist:** a 40-name/power-0.5 dry-run errored on 18/40 names, all
  "No tradable option contracts." Root cause (confirmed on AEE — lists only Jun 18 & Jul 17 monthlies):
  the probe window `[target−10, target+14]` = `[Jun 20, Jul 14]` for a 21-DTE target on Jun 9 fell in the
  GAP between the two monthlies. Liquid names survived because weeklies always land in-window; monthly-only
  mid-caps (the ones the flatter weighting surfaces) didn't.
- **Fix:** upper bound `target+14` → **`target+35`** so the window always spans a full monthly cycle
  (consecutive 3rd-Friday monthlies ≤35d apart). Lower bound kept at `target−10` so liquid names aren't
  pulled to an ultra-short weekly. Selection remains closest-to-target → widening only adds fallbacks.
- **Tests:** added `_select_expiry` coverage (previously only monkeypatched away): monthly-cycle window
  reach + closest-to-target pick. **327 pass.**
- **Re-ran the 40-name dry-run:** errors **18→3** (only BRK-B + PCAR/UDR, genuinely unlisted), sized
  signals **11→22**; recovered names trade the Jul 17 monthly, liquid names keep the Jul 2 weekly, and the
  book is slightly more two-sided (puts on AJG/BLDR/OTIS now appear). Note: HBAN funded 100× a $0.18 call
  — the deferred cheap-option contract-count cap (todo) is now more visible with mid-caps in scope.

### Session 10 (cont. 6) — cheap-option cap REJECTED (decision) + live run
- **Decision (user-chosen, logged in `docs/decisions.md`):** do NOT build a cheap-option/contract-count
  cap. Tails are the model's edge (the `decompose_edge` "shape edge"), not its weak spot; concentration
  is already bounded by per-name (5%) + gross (20%) caps and 40-name diversification (HBAN 100× = ~$1.8k);
  limit orders handle fills (no slippage-through-price); and Kelly oversizing on noisy tail probs is
  already covered by ¼-Kelly + max-fraction (the correct uniform lever, vs a structural anti-tail bias).
  Thesis-aligned alternative = a better tail model (fat-tail forecaster), backtest-gated. Todo item
  flipped to REJECTED. (Mirrors the block-bootstrap rejection: results logged so they aren't re-proposed.)
- **Live run** (`run_daily --live`, 40-name watchlist seed 739776 / power 0.5 / widened expiry, bankroll
  $84,593): 23 sized signals, 15 no-BUY, **2 errors** (BRK-B symbol-format + UDR unlisted; PCAR priced
  this time). 100%-bullish → 20% haircut → 16% gross budget; existing ~$9k premium left room for 2:
  **ABBV260702C00225000 ×2 — FILLED @ avg $6.75** (marketable, beat the $6.93 limit), and
  **HUM260702C00385000 ×5 @ $7.66 — SUBMITTED, resting `new`.** HUM again the cleanest edge (+11.6, drift 90%).
  Pre-flight confirmed no resting orders beforehand; entry guard auto-excluded held-call names (GOOGL).

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

## Session 12 — 2026-06-16 — GarchFhsEventForecaster (ACCEPTED; new production default)
- **Goal:** build and backtest a composed GARCH-FHS + event-conditioned forecaster.
- **Built:**
  - `forecast/garch_fhs_event_forecaster.py` — `GarchFhsEventForecaster`: normal days use GARCH-filtered
    z-draws scaled by current σ² and propagate the variance recursion forward; event days (earnings/FOMC)
    draw raw returns from the empirical event distribution (bypassing σ² scaling), then advance σ² from
    the shock. Key insight: earnings surprise magnitude is not a function of background vol — raw draws
    preserve this. After the shock, GARCH correctly spikes σ² for subsequent normal days.
  - `forecast/garch_fhs_event_forecaster.py` (cont.) — `GarchFhsEventFactory` in `factories.py`: same
    structure as `EventBootstrapFactory`; builds calendar + conditioned distributions + forward schedule;
    degrades to plain `GarchFhsForecaster` if the event feed fails.
  - `scripts/compare_forecasters.py` (prior context) — compare garch_fhs vs event_bootstrap per ticker.
  - `backtest/log_score.py` — `rolling_log_score_backtest_garch_fhs_event`: mirrors the conditioned
    backtest, builds `ReturnDistribution` from training slice for GARCH fitting at each holdout iteration.
  - `backtest/standardized_test.py` — `GARCH_FHS_EVENT_FORECASTER_NAME = "garch_fhs_events"`,
    `_run_garch_fhs_event_backtest`, updated validation and dispatch.
  - `scripts/run_standardized_test.py` — `--garch-fhs-events` CLI flag, updated headline logic.
  - `tests/test_garch_fhs_event_forecaster.py` — 8 tests: prices, normal-only vs plain FHS (bit-identical),
    event-day widens distribution, GARCH-state advance from event shock, horizon mismatch, reproducibility,
    input validation, factory degrades to plain FHS on broken event source.
- **Run:** `python scripts/run_standardized_test.py --events --garch-fhs --garch-fhs-events --horizon 21`
- **Results (h=21, 20-tkr, 860 forecasts each):**
  | Forecaster | mean log score | vs bootstrap |
  |---|---|---|
  | garch_fhs_events | −4.868 | +0.224, t=3.65, **p=0.000276** |
  | event_bootstrap | −4.892 | +0.200 |
  | garch_fhs | −4.959 | +0.107 |
  | bootstrap | −5.091 | — |
  garch_fhs_events win rate vs bootstrap: 65% (13/20). vs event_bootstrap: pooled +0.024, 13/20 win.
- **Decision:** ACCEPTED. garch_fhs_events passes the standing criterion (p=0.000276 vs bootstrap).
  `GarchFhsEventFactory` wired as the new production default in `_default_forecaster_factory()`.
  `EventBootstrapFactory` retained (reachable via `--events`; `--garch-fhs-events` explicit opt-in).
- **Tests:** +8 → **368 pass**. Artifacts: `output/standardized_test_2026-06-16_h21_hold63.{md,csv}`.
- **Open follow-ups:**
  - Formal paired t-test of garch_fhs_events vs event_bootstrap (direction clear, significance not computed).
  - h=5 run (the S9 leaderboard was at h=5; new leaderboard is h=21).
  - 40-ticker robustness run to mirror the GARCH-FHS S11 robustness check.
