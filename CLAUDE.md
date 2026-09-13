# options-trader — project memory (auto-loaded)

Standalone Python project: PDF-based call-option valuation → fractional Kelly sizing →
Alpaca paper execution. V1 = single ticker (META).

⚠️ This is NOT the quant-research project. A parent `/home/azureuser/CLAUDE.md` for that
other project loads via directory-walk — IGNORE it here; its build/tooling rules do not
apply to options-trader.

## Start-of-session checklist
1. Read `docs/status.md` (current build/test state + recommended next step) and
   `docs/todo.md` (the work board). These two are small by design — read them every session.
2. Tell the user: current build status, what was last worked on, recommended next step.
3. Do NOT re-read `docs/sessions.md` (full history) or `docs/architecture.md` unless the
   task needs that depth — that's what keeps session-start cost low.

## Commands
- Install: `pip install -e .` (repo root; venv at `./venv`)
- Test: `python -m pytest -q`
- Standardized backtest / lift measurement: `python scripts/run_standardized_test.py [--events] [--horizon N]`
- Credentials: `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` env vars

## Decision criterion (STANDING — every forecaster change)
A forecaster change ships ONLY if it raises mean **log score** in the rolling out-of-sample
backtest, paired vs the incumbent across the 20-ticker standardized set. Not intuition, not
in-sample fit, not theory. Record so far: block bootstrap **rejected** (p≈2e-4 worse), event
conditioning **accepted** (earnings: +0.032, p≈0.027; +auto-fetched FOMC: +0.039, p≈0.011 — still
accepted, FOMC didn't degrade it), intraday-price anchor **accepted** (+0.075, p≈8e-6), MA-blend anchor
**rejected** (−0.107, monotonically harmful), drift-dampening / uniform mean-reset toward spot
**rejected** (best κ ≈ +0.001, p≈0.6 at h=5 and h=21 — per-ticker signal/noise effects cancel),
GARCH-FHS (Filtered Historical Simulation) **accepted, plain/ungated** (40-tkr h=21 **+0.137, p≈2e-4,
57% win**); its vol-divergence gate **rejected** (monotonically harmful). **BUT (S16) GARCH's complexity
is NOT justified over event-bootstrap:** on the standardized set h=21, event_bootstrap +0.153 vs
bootstrap and garch_fhs_events +0.184 — head-to-head **garch_fhs_events − event_bootstrap = +0.031,
p=0.38 (NOT significant)**. ~83% of GARCH's lift is the EVENT conditioning, not the GARCH machinery
(EWMA bootstrap already tracks recent vol). ⇒ **event_bootstrap is now the production default
(`_default_forecaster_factory` switched GarchFhsFactory → EventBootstrapFactory, S16; GARCH-FHS still
available via `--garch-fhs-events`, pinned by a regression test).** New forecasters must clear this bar
to become a default. **Option-Implied Beta forecaster (S15 built, S16 backtested):
TWO-WINDOW plain + event three-ways settled it. vs bootstrap: VOL-STRESS REGIME ARTIFACT (W1 +0.046
p≈1e-12 / W2 ≈0 p=0.31; garch−boot sanity flips sign, confirming W1 is the flattering regime). vs the
event-GARCH-FHS production default: event-OIB beats it BOTH windows (+0.048 / +0.026, ~55–58% of tkrs)
— but a SOFT win (event-GARCH underperformed its +0.224 here, even losing to event-bootstrap in W2, so
partly GARCH weakness not OIB strength). ⇒ borderline-positive vs the real incumbent, NOT a clean robust
win; **default NOT flipped.** Live angles: vol-stress regime gate (biggest, robust edge), OIB+GARCH
ensemble, or a tie-breaker window where GARCH is healthy. See `docs/results.md` / `docs/status.md` A.**
Historical
EOD chains DO work for backtesting (`get_option_contracts(status=INACTIVE)` + `get_option_bars(Day)`,
data since Feb 2024). (Open: 2nd-window confirm for OIB then live-wire + head-to-head vs event GARCH-FHS;
live-default wiring of FHS into `factories.py`; event-conditioned FHS; clean FOMC ablation. P-measure
de-meaning + VRP variance deferred per user. See `docs/results.md` + `docs/status.md` step A.)

## Architecture (one line each — full map in `docs/architecture.md`)
- `data/` — `get_history → StockReturnTS`; `data/events/` — `Event` / `EventSource` / `EventCalendar`
- `forecast/` — `ReturnDistribution` (PURE data container), forecasters (bootstrap / block / gaussian / event), `conditioned_returns`, `factories` (injectable live forecaster wiring; event-conditioned is the production default)
- `backtest/` — `log_score` (rolling + event-conditioned), `standardized_test`
- `universe/` — frozen S&P 500 snapshot + cap-weighted sampler

**Design invariant:** `ReturnDistribution` stays a pure data container — sampling strategy lives
in forecasters, event logic in the event layer. (Why: `docs/decisions.md`.)

## Docs (read on demand — NOT auto-loaded; keep this file small)
- `docs/status.md` — current state + recommended next step (read every session)
- `docs/todo.md` — service-organized work board (read every session)
- `docs/architecture-overview.md` — holistic view (two loops, layer map, invariants, improvement backlog)
- `docs/runbook.md` — weekly run procedure + how to make a change (backtest-gated)
- `docs/architecture.md` — component/service map, key signatures, file map
- `docs/decisions.md` — locked architecture decisions (ADRs)
- `docs/results.md` — forecaster log-score leaderboard + head-to-heads
- `docs/issues.md` — known model limitations / open conceptual problems
- `docs/sessions.md` — chronological session log (history)

## Conventions
- Python 3.11+; numpy / pandas / scipy / yfinance / alpaca-py. Style: dataclasses, module
  docstrings, validation in constructors. Tests are synthetic / no-network.
- **At session end:** update `docs/status.md` + append `docs/sessions.md`; update `docs/todo.md`
  when items change; log any new forecaster result in `docs/results.md`.
