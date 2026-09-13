# Status

_As of 2026-09-13 (end of Session 19). Update this at every session end._

## Build
- Clean. **453 tests passing** (`python -m pytest -q`). Tests are synthetic / no-network; also
  end-to-end live-verified against real Alpaca/yfinance data this session (see Session 19).
- Install: `pip install -e .`; venv at `./venv`. No new deps pending.
- Git remote: `https://github.com/wiblert/options-trader` (public), pushed Session 19.

## Components
| Layer | State |
|---|---|
| `data/` (history + events) | ✅ complete |
| `forecast/` (ReturnDistribution, bootstrap/block/gaussian/event, conditioned_returns) | ✅ complete |
| `backtest/` (log_score rolling + conditioned, standardized_test) | ✅ complete |
| `universe/` (snapshot + cap-weighted sampler) | ✅ complete |
| `viz/` (charts) | ✅ complete |
| `valuation/` (black_scholes, option_valuer) | ✅ complete (Session 5; held-to-expiry V1) |
| `data/options_chain.py` (Alpaca contracts + chain) | ✅ complete (Session 5) |
| `sizing/kelly.py` (fractional Kelly + cap) | ✅ complete (Session 5) |
| `execution/broker.py` (Alpaca paper orders) | ✅ complete (Session 5; live read + dry-run verified) |
| `run_daily.py` (ENTRY orchestrator + CLI) | ✅ complete (Session 5; 15-ticker dry-run verified) |
| `portfolio/constructor.py` (gross + directional + per-name caps) | ✅ complete (Session 6) |
| `manage_positions.py` (EXIT pass: reprice holds → sell-to-close) | ✅ complete (Session 10; live sell verified) |
| `data/occ.py` (OCC option-symbol parser) | ✅ complete (Session 10) |
| `execution/broker.py` `submit_sell` (sell-to-close limit) | ✅ complete (Session 10) |
| `webapp/` (forecast diagnostic web UI) | ✅ complete (Session 11; live-verified vs Alpaca) |
| `webapp/` review app (human-in-the-loop run_daily: review→approve→execute) | ✅ complete (Session 14; live-verified vs Alpaca paper) |
| `forecast/beta.py` + `breeden_litzenberger.py` + `option_implied_beta_forecaster.py` + `data/options_history.py` | ✅ built (S15), **2-window plain + event 3-way backtested S16**. vs bootstrap: regime artifact (W1 +0.046 / W2 ≈0). vs event-GARCH-FHS default: beats it BOTH windows (+0.048/+0.026) but soft (GARCH weak here). **NOT flipped** — vol-stress-conditional / ensemble candidate |
| `backtest/option_implied_beta_backtest.py` (as-of index PDF + rolling backtest) | ✅ complete (Session 16) |
| `forecast/option_implied_forecaster.py` + `OptionImpliedFactory` + `backtest/option_implied_backtest.py` + `scripts/run_option_implied_backtest.py` | ✅ built (S17) — **BENCHMARK ONLY** (own-options risk-neutral PDF; reproduces the prices for sale → circular to trade; NEVER a default). Head-to-head vs event-bootstrap measures whether our forecast beats the market price. |

- **Production default forecaster:** **50/50 `BlendFactory`** = event-bootstrap ⊕ event-conditioned OIB —
  **switched S18 (per user direction)** from `EventBootstrapFactory`. Path-level probability mixture: half
  the terminal-price mass from the event-conditioned bootstrap (historical), half from event-OIB (forward
  option-implied SPY/IWM PDF · Beta + earnings-conditioned idiosyncratic). S18 backtest: blend beats
  event-bootstrap +0.0107 (p≈6e-14, 27/40, 1yr) — small, OIB-inherited, single-window; shipped as the
  more robust expression of the signal, not on strong evidence. Live dry-run verified (`run_daily
  --tickers AAPL`). Each arm degrades to plain bootstrap on feed failure; `--no-events` → plain bootstrap,
  `--garch-fhs` → GARCH-FHS. Regression test: `test_default_forecaster_factory_is_blend`. **Open gate:**
  single-window edge — held-out 2nd window + vol/beta router (to-do) still pending. Prior default
  `EventBootstrapFactory` (S16, +0.153 vs plain bootstrap) remains a sub-factory of the blend.
- **Event sources (live):** yfinance earnings + **`FomcCalendarSource`** (auto-fetched Fed meetings)
  + manual macro CSV. FOMC is no longer hand-maintained.

## Recommended next step
_Production default is now the **50/50 blend** (event-bootstrap ⊕ event-OIB), switched S18 per user
direction. The Architecture / tech-debt cleanup section in `docs/todo.md` is now fully resolved
(Session 19). The highest-priority follow-ups, in order:_

**A. Validate the blend default properly** (it shipped on a single-window edge). Gate it on a
**held-out 2nd window** + build the **vol/beta router** (`docs/todo.md` ensemble item) — item 1 showed
the OIB-vs-bootstrap edge is cleanly cross-sectional (high-vol/beta vs defensive), so a router may beat
the flat 50/50 blend. Also open: **edge-aware watchlist sampling** (success-weighted / low-volume tilt,
new Universe to-do).

**B. ⚠️ get the review web app to render in the browser** (Session 14 unfinished). The app
(`/review`) is code-complete and the server + API are proven working via curl, but it shows a blank
page / spinner when viewed over the VS Code SSH port-forward. Vendoring Plotly locally did not fix it.
Diagnose from the browser side: DevTools Console/Network on the blank page, confirm the forwarded local
port, and `curl http://localhost:<port>/review` from the laptop. See the OPEN note under "Last worked on".

Below this: ENTRY (`run_daily`) and now EXIT (`manage_positions`) are both live-verified. The remaining gaps
before a real unattended daily run, in priority order:
1. **Fill/position reconciliation** — after placing, poll order status, confirm fills via
   `get_positions`, persist what we hold. Decide DAY vs GTC. (As of S10 two orders rest: the AVGO
   405 BUY from earlier filled; the META put SELL is resting `new` @ $5.23.)
2. **Idempotency / re-run guard — open-orders awareness DONE (S10 cont.3); `client_order_id` still open.**
   `broker.get_open_orders()` now feeds both passes: a resting BUY counts toward the entry one-call/
   one-put-per-name guard, and a resting SELL blocks the exit pass from resubmitting a sell on that
   symbol. So a same-day re-run no longer double-submits via the order book. STILL OPEN: a deterministic
   `client_order_id` (day-scoped) as the broker-enforced backstop, and the race window between submit and
   the order appearing as "open".
3. **Exit logic — partially DONE (S10).** `manage_positions.py` reprices holds and sells when fair
   value is ≥`sell_threshold` (default 15%) below the bid. Still open: expiry handling (no rule for
   options nearing/at expiry beyond a SKIP), partial-close sizing (sells the full position), and a
   time-stop / max-hold rule. Exit is purely value-driven today.
4. **Market-hours awareness** — neither pass checks the clock; DAY orders placed while closed rest
   to next open (fine, but should be explicit).
5. **Cheap-option contract limit** — Kelly buys huge counts of cheap OTM options (MNST 31, GOOGL 5+)
   where tails are weakest; add a per-contract/notional sanity cap.
6. **Edge quality** — runs remain dominated by ISSUE-1 drift (drift% mostly high, several >100%).
   Fundamentals anchor + fat-tail forecaster are the real-edge work (backtest-gated).

Off the execution path (forecaster quality, gated by the log-score criterion): **fat-tail
forecaster** (Student-t / jump diffusion) remains the highest-value open forecaster experiment
(block and Gaussian both ≈ bootstrap, so tails are the remaining gap).

Full backlog: `docs/todo.md`.

## Last worked on
**Session 19 (2026-09-13) — GitHub remote + full architecture/tech-debt cleanup.**
Pushed the project to `https://github.com/wiblert/options-trader` (public repo, initial commit +
description). Then worked through the entire S18 "Architecture / tech-debt cleanup" section in
`docs/todo.md` end to end (10 items, all resolved — see todo.md for per-item detail): centralized the
remaining `risk_free_rate`/`max_fraction` drift into `config.py`; renamed the misleading `index`-named
helpers (`build_eod_index_pdf`→`build_eod_pdf`, `HistoricalIndexPdf`→`HistoricalEodPdf`,
`live_index_pdf`→`live_eod_pdf`); added `LiveEodPdfCache` so the blend's OIB arm fetches SPY/IWM chains
once per watchlist run instead of once per ticker; clarified the architecture.md /
architecture-overview.md split in both headers; hardened `rolling_log_score_backtest` against a
mutating factory (`copy.deepcopy(rd)`); deduped the double option-chain fetch in `plan_ticker`/
`_select_expiry`; parallelized per-ticker planning in `run_daily` via `ThreadPoolExecutor` (`--workers`,
default 8); added `_ticker_seed` so MC noise is decorrelated across tickers instead of sharing seed=42;
collapsed the duplicate `OptionValuation.expected_return_buy` field; and split `docs/sessions.md`
Sessions 1–9 into `docs/sessions-archive.md`. All were mechanical/structural — no behavior change.
+2 tests (`LiveEodPdfCache`) → 452 pass.
**Then ran a live end-to-end integration test** (`run_daily` dry-run against real Alpaca/yfinance,
no `--live`) and it caught a real bug the unit tests missed: `LiveEodPdfCache`'s check-then-fetch-
then-store logic is defeated by the NEW thread-pooled planning — verified live, 4 tickers produced
4x SPY + 4x IWM chain fetches instead of the intended 1x each. Fixed to single-flight per-key locking
(the lock is held across the whole cache miss, so concurrent requests for the same key block and
reuse the result instead of all fetching). Re-verified live: 4 tickers → exactly 1x SPY + 1x IWM.
Added a concurrency regression test (8 threads behind a start barrier; fails 8-vs-1 against the old
logic, passes 1-vs-1 against the fix) — the earlier sequential-only test couldn't have caught this.
+1 test → **453 pass.** Full CLI (`python -m options_trader.run_daily --tickers ...`) run end-to-end
dry-run multiple times post-fix: EXIT pass reviewed real held positions (AXP/MRNA/TRGP), ENTRY pass
planned 2–4 tickers concurrently with correct sizing/caps output, zero errors.

**Session 18 (2026-06-23) — 50/50 blend forecaster (event-bootstrap ⊕ event-OIB): ensemble to-do (d).**
Built the path-level 50/50 probability mixture: `forecast/blend_forecaster.py`
(`BlendForecaster` + pure `blend_price_distributions` — stacks both forecasters' weighted price samples,
halves each side's weights → mixture sums to 1, preserves each shape), `factories.py::BlendFactory`
(NOT a default), `backtest/blend_backtest.py::rolling_log_score_backtest_blend_event` (scores eboot/oib/
blend in lockstep, components match their standalone backtests), `scripts/run_blend_backtest.py`
(`--mode vs-eventboot` items 1+3a, `--mode vs-options` item 3b). +10 tests → **450 pass.**
**Results (1yr, h=21, holdout=252):** blend beats event-bootstrap **+0.0107, p≈6e-14, 27/40 names**
(40-tkr) — but does NOT beat OIB alone (−0.002, n.s.); gain is ~85% inherited from OIB, blend is a
robustness play (sits between both arms on 33/40). vs own-options (20 tkr): +2.16 pooled but thin-chain
inflated (liquid-only +0.73, median −0.026, win 47% → tail-driven edge, not typical-day). 50/50 chosen
over learned weights to avoid overfitting the combiner. **Default FLIPPED to the blend per user
direction** (`_default_forecaster_factory` → `BlendFactory`); to match the backtest the LIVE
`OptionImpliedBetaFactory` gained an optional `event_source` (event-conditioned idiosyncratic). Live
`run_daily --tickers AAPL` dry-run verified. Regression test → `test_default_forecaster_factory_is_blend`.
**Open risk:** single-window edge; held-out 2nd window + vol/beta router are the proper gate. Full
numbers: `docs/results.md`; ADR: `docs/decisions.md`; charts `output/blend_compare.png`,
`output/blend_vs_options.png`.

**Session 17 (2026-06-23) — Option-Implied (own-options) BENCHMARK forecaster (accuracy vs the market price).**
Built the "is our forecast more accurate than the prices for sale?" check the user asked for. New
`forecast/option_implied_forecaster.py::OptionImpliedForecaster` reconstructs a ticker's OWN
Breeden-Litzenberger risk-neutral PDF and re-anchors it to spot (deterministic — the PDF grid IS the
distribution; `n_paths`/`seed` accepted only for constructor uniformity). Reuses existing machinery
wholesale: `build_eod_index_pdf` and `live_index_pdf` are both symbol-generic. Added
`OptionImpliedFactory` (live, NOT a default — `run_daily` default stays `EventBootstrapFactory`),
`backtest/option_implied_backtest.py` (`rolling_log_score_backtest_option_implied` +
`HistoricalTickerPdf` cache, skips+counts dates with no PDF), and
`scripts/run_option_implied_backtest.py` (paired head-to-head vs the event-bootstrap production
default; NEGATIVE (option_implied − event_bootstrap) ⇒ our forecast beats the market price = edge).
**⚠️ BENCHMARK ONLY, never a production default** — it reproduces the prices already for sale, so
valuing those same options against it is circular (ADR in `docs/decisions.md`; `docs/issues.md` ISSUE-N).
+10 synthetic tests → **440 pass**. **Live smoke (network) verified end to end** vs Alpaca historical
chains: META, 2024-06→2025-03, h=21 holdout=30, n=10 paired, 0 skipped → option_implied −5.688 vs
event_bootstrap −5.453, **diff −0.235 (t=−2.0, p=0.075)** — illustrative (n=10, one ticker, not a gated
claim), but confirms the full pipeline (chain discovery → EOD bars → BL PDF → forecast → scoring →
paired report). Artifact: `output/option_implied_backtest_smoke.md`.
**Then ran the 30-ticker / 1.5-year quantile analysis** (`scripts/analyze_option_implied_quantiles.py`,
n=4530 evals, 27/30 tickers): mean advantage +1.52 but win rate 46.3% — a **FAT-TAIL edge, U-shaped in
realized return**. Best quantile = top return decile (big up-moves +12%..+53%, median adv +1.32, 80%
win); bulk (small everyday moves) has negative-median advantage (market PDF better-calibrated there).
Full table + caveats in `docs/results.md`; chart `output/option_implied_quantiles.png`.

## Older sessions
Session 16 cont.4 and earlier (S1–S16) live in `docs/sessions.md` (full chronological history) —
trimmed from here in S18 to keep this doc small per the start-of-session contract. Sessions 1–9 were
further split out of `docs/sessions.md` itself into `docs/sessions-archive.md` (S18 tech-debt cleanup)
to keep that file's read cost low too. Decisions in `docs/decisions.md`, forecaster numbers in
`docs/results.md`.
