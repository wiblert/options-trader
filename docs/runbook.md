# Runbook

How to operate options-trader and how to change it safely. Architecture context:
`docs/architecture-overview.md`. Decision rules: `docs/decisions.md`.

---

## A. Weekly run — every Monday

Run the pipeline weekly to refresh option positions. `run_daily` now executes both passes:
the **exit pass** (reprices held contracts, sells overpriced ones) runs first, then the
**entry pass** (opens new positions). Treat the live step as **semi-manual**: dry-run,
eyeball, then submit, then check fills by hand.

### Preconditions
- `cd ~/options-trader && source venv/bin/activate` (or `pip install -e .` if deps changed).
- Credentials exported: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (paper account).
- Run during US market hours (09:30–16:00 ET) so live prices anchor on the real tape and limit
  orders can fill the same session. Off-hours, DAY orders rest to next open and the forecast
  silently falls back to the last close (logged as `spot_source = last_close`).
- Tests green: `python -m pytest -q`.

### Step 1 — dry run (no orders)
```bash
python -m options_trader.run_daily --tickers META,GOOGL,AVGO,...   # or omit --tickers for a sampled watchlist
# knobs: --n-tickers 40  --target-dte 21  --bankroll <$>  --kelly-fraction 0.25
#        --max-fraction 0.05  --n-paths 50000  --seed 42 (forecaster paths)
#        --watchlist-seed <n>  --power 0.5   (ticker sampling; see below)
#        --sell-threshold 0.15              (exit: sell if fair_value is >=15% below bid)
#        --no-manage                        (skip exit pass; entry only)
```
The command runs **two passes** in sequence:
1. **EXIT pass** — reprices every held option contract and submits sell-to-close limit orders at the
   bid for any where `(bid − fair_value) / bid ≥ sell_threshold`. Skip with `--no-manage` if you only
   want new entries. Summary table shows: `symbol`, `bid`, `fair`, `gap`, `action`, `order`.
2. **ENTRY pass** — plans the watchlist, allocates under portfolio caps, submits buy orders.

The sampled watchlist (no `--tickers`) draws **40** names from the frozen S&P 500 snapshot,
cap-weighted with `power=0.5` (∝ √cap — flattens the mega-cap tilt so mid-caps surface), and the
sampling seed **rotates daily** (run-date ordinal): fresh names each calendar day, stable within a
day so a re-run sees the same list. Pin it with `--watchlist-seed <n>` to reproduce a given day, set
`--power 1.0` for pure cap-weighting or `0.0` for uniform. Per-ticker guard: at most one call + one
put per name (counting existing holdings + resting buys).
Read the entry summary table: per ticker the `contract`, `ask`, `edge`, **`ev`** (event days in the
horizon — a non-empty value means an earnings/macro event is priced into that forecast), **`drift%`**
(share of the edge that is the distrusted momentum drift — high drift% = low conviction), sized
`N`/`cost`, the `alloc` verdict (✓ fund / ✗ reason), and the directional split + any unhedged haircut.
The forecaster defaults (S18) to the **50/50 blend** of event-conditioned bootstrap ⊕ event-conditioned
Option-Implied Beta. Both arms use the live event calendar — yfinance earnings + auto-fetched FOMC/Fed
meetings (`FomcCalendarSource`, no upkeep) + `config/manual_events.csv` for ad-hoc macro/FDA — and the
OIB arm also pulls the SPY/IWM risk-neutral PDF. Pass `--no-events` to fall back to the plain bootstrap,
`--garch-fhs` for GARCH-FHS. ⚠️ The blend shipped on a single-window edge (see `docs/status.md`).

### Step 2 — sanity checks before going live
- **Idempotency:** the exit pass guards against double-selling (checks for resting SELL orders before
  resubmitting). The entry pass checks resting BUY orders + current positions. A re-run the same day
  is safe as long as prior orders haven't filled.
- Is the book wildly one-sided? (drift-dominated sweeps are ISSUE-1, not 9 independent edges.)
- Are `drift%` values plausibly < ~70% on the names you'd actually fund?

### Step 3 — go live
```bash
python -m options_trader.run_daily --tickers <same set> --live
```
Exit sells are limit-at-bid; entry buys are marketable limit-DAY at the ask. `order` column shows
SUBMITTED / DRY_RUN / SKIPPED / REJECTED.

### Step 4 — reconcile (manual, until P0 lands)
- Re-check `get_positions` / Alpaca dashboard: did the SUBMITTED orders fill or are they resting?
- Note what was opened (there is no persisted run log yet — record it in `docs/sessions.md`).

### Refreshing the test universe (rare, deliberate)
The S&P 500 snapshot is frozen and committed. Only refresh when you intend to change the test set:
`python scripts/run_standardized_test.py --refresh-universe`.

---

## B. Making a change

Two cases. Anything touching the **forecast distribution** is gated by the backtest; everything
else (execution, sizing caps, plumbing) is gated by tests + a dry run.

### B1 — Changing a forecaster / the forecast distribution  ← the gated path
> A forecaster change ships ONLY if it raises mean **log score** in the rolling out-of-sample
> backtest, paired vs the incumbent across the 20-ticker standardized set. Not intuition, not
> in-sample fit, not theory. (Standing rule — `docs/decisions.md`.)

1. **Implement** behind the existing seams: a new `Forecaster` subclass (keep the
   `forecast(horizon_days, spot)` signature; put any new sampling logic in the forecaster, not in
   `ReturnDistribution`), or a new spot-anchor in `data/spot_anchor.py` + `backtest/anchor.py`.
2. **Unit tests** (synthetic, no-network) — `python -m pytest -q`. Mirror the moment/seed/edge-case
   tests in `tests/test_*_forecaster.py`.
3. **Register it** for the benchmark: add the class to `FORECASTER_CLASSES` in
   `backtest/standardized_test.py` (uniform `(distribution, n_paths, seed)` constructor), or add an
   experiment runner like `run_anchor_experiment` for non-forecaster knobs.
4. **Backtest, paired vs the incumbent:**
   ```bash
   python scripts/run_standardized_test.py                 # forecasters
   python scripts/run_standardized_test.py --events        # adds event_bootstrap
   python scripts/run_standardized_test.py --anchor-experiment   # spot-anchor head-to-head
   ```
   Read the headline paired t-test (mean log-score diff, p-value, per-ticker win rate). Smoke first
   with `--n-sampled 3 --n-paths 3000` to fail fast.
5. **Decide by the gate.** Accept only on a positive mean log-score diff (and you'll want
   significance, p<0.05). Record the head-to-head in `docs/results.md` and the verdict in the
   CLAUDE.md decision-criterion record — **whether accepted or rejected** (rejections are results too:
   block bootstrap is logged as rejected).
6. **If accepted, wire it into production.** The forecaster is injectable: add a factory in
   `forecast/factories.py` (or a new `Forecaster` subclass) and set it as the default in
   `run_daily._default_forecaster_factory`, or pass `forecaster_factory=` to `run_daily`. The
   event-conditioned forecaster and the intraday anchor are already the live defaults this way.

### B2 — Changing execution / sizing / portfolio / plumbing
Not distribution-affecting, so the backtest gate doesn't apply — but the live blast radius is larger.
1. Implement; keep the broker paper-only and dry-run-default.
2. `python -m pytest -q` — these layers have direct unit tests (`test_kelly`, `test_broker`,
   `test_portfolio_constructor`, `test_run_daily`). Tests are mocked/no-network.
3. **Dry-run the full pipeline** (`run_daily` without `--live`) and confirm the summary looks right
   before any live run.

### Always, at session end (per CLAUDE.md)
Update `docs/status.md` (state + recommended next step) and append `docs/sessions.md`; update
`docs/todo.md` if items changed; log any forecaster result in `docs/results.md`.
