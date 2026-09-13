# Architecture Overview

Holistic view of options-trader: what it does, the two loops, the ONE layers
summary table, and the cross-cutting invariants + improvement backlog.

**Split with `docs/architecture.md`:** that doc goes one level deeper — per-module
function signatures for the layers complex enough to need it (data/, forecast/,
backtest/, universe/) — and does not repeat this file's Layers table. For how to
operate the system see `docs/runbook.md`; for *why* decisions were made see
`docs/decisions.md`.

---

## What it does, in one paragraph

Given a ticker, the system forecasts a **probability distribution of the underlying's
price at an option's expiry** (a 50/50 blend of an event-conditioned historical-return bootstrap and a
forward option-implied view — the SPY/IWM risk-neutral PDF mapped on by Beta),
values every contract in the chain as the **discounted expected payoff under that forecast**
(the real-world / P-measure EV), compares it to the **market quote** to find an edge, sizes
the favorable ones with **fractional Kelly**, applies **portfolio-level risk caps**, and
submits **paper limit orders** to Alpaca. V1 is long-only, held-to-expiry, entry-only.

The thesis: the market prices options off implied vol (a Q-measure, no-arb price); we price
them off our own forecast of where the stock will actually go (a P-measure EV). We buy when
our EV exceeds the ask. Whether that forecast is any good is decided **empirically by a
log-score backtest**, never by intuition.

---

## The two loops

The codebase is really two loops that meet at one seam (the forecaster).

```
   ┌─────────────────────── RESEARCH LOOP (offline, backtest-gated) ───────────────────────┐
   │                                                                                        │
   │  universe sampler → get_history → ReturnDistribution → [forecaster] → PriceDistribution│
   │                                          │                                  │          │
   │                                   (+ EventCalendar)                   rolling log-score │
   │                                                                              │          │
   │                                                              standardized_test (20 tkr) │
   │                                                              paired t-test vs incumbent │
   │                                                                              │          │
   │                                                        ACCEPT iff mean log-score lifts  │
   └──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                                   │  (an accepted forecaster becomes a default)
                                                   ▼
   ┌─────────────────────── TRADING LOOP (weekly, live paper) ──────────────────────────────┐
   │                                                                                         │
   │  watchlist → plan_ticker ─┐                                                             │
   │      get_history          │                                                             │
   │      spot_anchor ─────────┤→ [forecaster].forecast(horizon, spot) → PriceDistribution   │
   │      _select_expiry       │                          │                                  │
   │      get_option_chain ────┘                          ▼                                  │
   │                                          OptionValuer.value → OptionValuation (edge)     │
   │                                                      │                                  │
   │                                          KellySizer.size → KellySizing (n_contracts)     │
   │                                                      │                                  │
   │           (all tickers) → PortfolioConstructor.allocate → gross/hedge/per-name caps      │
   │                                                      │                                  │
   │                                          AlpacaBroker.execute → paper limit order        │
   └─────────────────────────────────────────────────────────────────────────────────────────┘
```

**The seam is the forecaster.** Same `Forecaster.forecast(horizon_days, spot) -> PriceDistribution`
contract is exercised by both loops, and both now build it through an injectable factory — so the
forecaster the research loop validates is the one the trading loop runs (closed in Session 9).

---

## Layers & responsibilities

| Layer | Package | Input → Output | Owns |
|---|---|---|---|
| Market data | `data/history.py`, `data/options_chain.py`, `data/spot_anchor.py` | ticker → `StockReturnTS`, `[OptionContract]`, anchor spot | Alpaca/yfinance I/O, typed errors, timeouts, source fallback |
| Events | `data/events/` | source → `EventCalendar` | event→return-index mapping, BMO/AMC timing |
| Return dist. | `forecast/return_distribution.py`, `conditioned_returns.py` | log-returns → weighted empirical dist(s) | **pure** sample container; decay weights; event partition |
| Forecaster | `forecast/*_forecaster.py` (+ `base.py`) | `ReturnDistribution`/index PDFs, `spot`, `horizon` → `PriceDistribution` | sampling strategy (iid / block / gaussian / event-routed / GARCH-FHS / option-implied-beta / own-options benchmark / **N-way blend mixture**). Production default = **50/50 event-bootstrap ⊕ event-OIB blend** (S18) |
| Valuation | `valuation/option_valuer.py`, `black_scholes.py` | `PriceDistribution` + `OptionContract` → `OptionValuation` | P-measure EV, edge vs quote, BUY/SELL/HOLD, drift/shape split |
| Sizing | `sizing/kelly.py` | `OptionValuation` + bankroll → `KellySizing` | full-Kelly f*, ¼-Kelly, per-position cap, integer contracts |
| Portfolio | `portfolio/constructor.py` | `[Candidate]` → `AllocationResult` | gross-premium cap, unhedged haircut, per-name cap, greedy alloc |
| Execution | `execution/broker.py` | `KellySizing` → `OrderResult` | paper-only limit orders, account guards, dry-run |
| Orchestration | `run_daily.py` | watchlist → `DailyRunResult` | wires the trading loop, per-ticker isolation, summary |
| Evaluation | `backtest/` | `StockReturnTS` + forecaster factory → `BacktestResult` | rolling log-score, PIT, standardized 20-ticker paired test |
| Universe | `universe/` | frozen S&P 500 snapshot → sampled test set | reproducible cap-weighted sampling |

Each arrow is a typed, frozen dataclass — the layers are decoupled through data, not calls into
each other's internals.

---

## Cross-cutting invariants (the load-bearing rules)

1. **`ReturnDistribution` is a pure data container.** Sampling strategy lives in forecasters,
   event logic in the event layer, anchor choice in a spot-source. Never push behavior into it.
2. **Held-to-expiry consistency is the caller's contract.** The `PriceDistribution` must be
   forecast at exactly the option's expiry horizon; the valuer and sizer assume this and don't
   re-check it. `run_daily` enforces it via `_select_expiry` → `horizon`.
3. **Injectable sources over hardcoded choices.** Spot anchor (`SpotAnchor`) and forecaster
   factory are/should be parameters, so the backtest and production exercise the *same* code with
   different wiring. (`spot_anchor` done; forecaster factory is the open one.)
4. **Typed errors, fail-soft in batch.** `CredentialsError` always propagates (never silently
   demote on auth); data-unavailable is typed; per-ticker planning failures are isolated so one
   bad name never aborts the weekly run.
5. **Paper-only, limit-only, dry-run-default.** The broker cannot reach live trading; every order
   is a marketable limit; the CLI requires explicit `--live`.
6. **The backtest gate is sacred.** No forecaster change becomes a default without a paired
   log-score lift (`docs/decisions.md`). Record so far: block ❌, event-bootstrap ✅, intraday anchor ✅,
   drift-dampening ❌, GARCH-FHS ✅, GARCH-over-event-bootstrap ❌ (insignificant), option-implied-beta
   ~ (borderline/regime-dependent), **50/50 blend ✅ (S18 — but small, single-window; shipped per user as
   the robust expression of the OIB signal, held-out 2nd window still pending)**. own-options forecaster
   = benchmark only (never a default — circular).

---

## Where validation meets production (gap closed, Session 9)

Both loops build the forecaster through a factory, so the validated forecaster trades:

- **Forecaster is injectable** — `forecaster_factory` on `plan_ticker`/`run_daily` (`forecast/factories.py`:
  `ForecastContext` + the factory family), mirroring the backtest's factory and the spot-anchor pattern.
- **The production default is the 50/50 blend** (`BlendFactory` of `EventBootstrapFactory` ⊕
  event-conditioned `OptionImpliedBetaFactory`, S18). `run_daily` builds a live `EventCalendar`
  (yfinance earnings + auto-fetched FOMC + manual macro CSV) per ticker and a **forward event schedule**
  (`EventCalendar.forward_schedule`); the OIB arm additionally pulls the SPY/IWM risk-neutral PDF
  (`live_eod_pdf`) and conditions its idiosyncratic residuals on earnings. Each arm degrades to plain
  bootstrap on feed failure (or globally via `--no-events`); `--garch-fhs` selects GARCH-FHS instead.
- The validated **intraday anchor** is likewise live (`live_else_close_anchor`).

Remaining gap of this kind: none on the forecaster seam. ⚠️ The blend shipped on a single-window edge —
a held-out 2nd window + vol/beta router are the proper validation (`docs/todo.md`). Open forecaster work
(fat-tail, fundamentals anchor) is new edge, not unwired validated edge.

---

## Improvement backlog (architecture)

Prioritized. P0 = do before trusting an unattended weekly run; P1 = correctness/quality of the edge;
P2 = scale/cleanup.

**P0 — operational safety (entry-only system run on a schedule)**
- **Idempotency guard.** `execute()` never sets a `client_order_id`; a second Monday run
  double-submits. Derive `client_order_id` from `run_date + contract.symbol` and/or check open
  orders/positions before submitting. (`submit_buy` already accepts the param — just wire it.)
- **Fill / position reconciliation + run persistence.** Nothing records what was planned, ordered,
  filled, or held across runs (state is whatever Alpaca returns live). Persist a per-run record
  (JSON/CSV under `output/`) and poll order status after submit; decide DAY vs GTC.
- **Exit / expiry handling.** Entry-only: the book only grows. Define hold-to-expiry vs early-exit
  and add a sell path before running weekly for real.

**P1 — edge quality (all backtest-gated)**
- ✅ **DONE (S9): forecaster is an injectable factory; validate→trade gap closed.** Live default has since
  evolved S12 GARCH-FHS → S16 event-bootstrap → **S18 50/50 blend (event-bootstrap ⊕ event-OIB)**.
  ⚠️ The blend shipped on a single-window edge — held-out 2nd window + vol/beta router still pending.
- ✅ **DONE: architecture/tech-debt cleanup is now a dedicated `docs/todo.md` section** (S18); the P2
  list below is consolidated there.
- **Fat-tail forecaster** (Student-t / jump-diffusion) — the standing top forecaster experiment
  (block ≈ gaussian ≈ bootstrap ⇒ the gap is tails, not dynamics).
- **Fundamentals/fair-value anchor** (ISSUE-1) — the live edge is currently a noisy momentum drift;
  blend toward a valuation anchor and measure it.

**P2 — scale & cleanup**
- **Parallelize per-ticker planning.** Tickers are independent but planned sequentially with ~4
  sequential network calls each (history + latest price + 2 chain pulls). A thread pool turns a
  multi-minute 15-ticker run into seconds.
- **Dedupe the double chain fetch.** `_select_expiry` and `plan_ticker` both call
  `get_option_chain`; fetch once and reuse.
- **Centralize config.** `DEFAULT_N_PATHS` differs between `run_daily` (50k) and the backtest (10k);
  `max_fraction` defaults differ between `kelly.py` (0.20) and `run_daily`/CLI (0.05). Put run
  parameters in one typed config so the weekly run is reproducible and the knobs are discoverable.
- **Symbol-format mapping** (`BRK-B` → `BRK.B`) — a universe→Alpaca normalization layer; currently
  fails both history and chain for dual-class names.
- **Minor:** `OptionValuation.edge_pct_buy` and `expected_return_buy` are algebraically identical
  (`fair/ask − 1`) — collapse to one. Per-ticker forecaster seed is fixed at 42 (MC noise correlated
  across names); bump per ticker if independence matters.
