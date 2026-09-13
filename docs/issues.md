# Issues

Known limitations and open conceptual problems with the current model. Distinct from
`docs/todo.md` (concrete work items) — these are model/validity concerns that shape what the
pipeline can and cannot honestly claim. Every proposed fix is still gated by the standing
**log-score** decision criterion (CLAUDE.md) before it changes a default.

---

## ISSUE-1 — The forecast is a pure long-trend extrapolation; no sense of true valuation
- **Status:** open. Surfaced Session 5 from the first live forecast-vs-market comparison (META,
  2026-06-18 expiry).
- **Symptom:** the valuer flagged **SELL on all 24 calls and BUY on all 24 puts** — a perfectly
  one-sided sweep, zero HOLD. That is NOT 48 independent edges; it is a single directional view
  replicated across strikes.
- **Mechanism:** our forecast terminal mean was **597.81**, sitting ~$3.45 **below** the risk-neutral
  forward (**601.26** = spot grown at r). When the whole forecast distribution is shifted below where
  the market centers its, every call looks rich and every put looks cheap, monotonically with
  moneyness. The shift comes entirely from the bootstrap inheriting the **recency-weighted historical
  mean daily log-return** (≈ −9% annualized drift at the time) — i.e. the model is extrapolating
  recent price weakness and nothing else.
- **Root limitation:** the model has **no estimate of what the ticker is actually worth.** There is
  no fundamental anchor, no target/intrinsic value the price is expected to drive toward, and hence
  no mean-reversion to a fair value — only raw extrapolation of past returns. Historical-mean drift
  is also among the noisiest estimates in finance (huge standard error), so betting on it directly is
  fragile. A trustworthy options edge usually comes from a **volatility/shape** disagreement (which
  would produce BUY-some / SELL-some), not a clean directional sweep. (The market's IV smile —
  ~33% ATM rising to ~36–38% in the wings — is real curvature our flat historical bootstrap does not
  reproduce, confirming the current forecaster expresses level/drift, not vol structure.)
- **Proposed direction:** blend the historical-returns forecast with a **fundamentals-based forecast**
  — an estimate of fair/target value the price should converge toward — so the distribution has a
  valuation anchor rather than pure momentum. **Must be measured via the rolling log-score backtest
  before adoption** (per the decision criterion); not shipped on theory. Backlog item under
  Forecasters in `docs/todo.md`.
- **Interim mitigations to weigh:**
  - **De-mean onto the forward** to isolate any pure vol/shape edge (trade shape, not drift).
  - **Fractional Kelly + a guard** in the sizing layer so a noisy drift estimate cannot size into
    48 correlated bets that are really one bet. **DONE (Session 6):** `portfolio/constructor.py` adds a
    gross cap + a **directional-balance** cap (≤60%/side) — chosen over a correlation estimate to bound
    market beta directly. This limits how far the drift can tilt the whole book one way.
  - The backlogged **fat-tail forecaster** would give a genuine shape edge rather than a drift bet.

- **DECISION (Session 5): defer. Ignore the mean-drift problem for now** and stay focused on the
  end-to-end goal (value + buy a paper call). Revisit later — it is real and worth fixing, but it
  does not block execution.

- **Now measurable per signal:** `OptionValuer.decompose_edge` (Session 5) splits each contract's edge
  into `total = drift + shape` by re-centering our distribution on the forward. Confirms the pattern:
  drift is the directional tilt (helps puts / hurts calls, monotonic in strike); a non-drift "shape"
  residual (dispersion/tails + half-spread, mild long-vol lean) is the rest. The purchased META 570
  put was 43% drift / 57% shape. Use this to gate real conviction: a signal that is mostly drift is
  mostly the thing we don't trust.

- **Nuance worth preserving for the revisit (the mean-reset / shrinkage discussion):**
  - **Log-score barely sees the mean at short horizons.** Drift contributes `μ·h` to the terminal
    log-return; dispersion contributes `σ·√h`. For META today (σ≈2.25%/day, drift≈−0.037%/day,
    h=12): |drift|/(σ√h) ≈ **5.7%** — the center is off by ~1/20 of a std. So the alarming mean drop
    is distributionally almost invisible; it shows up only as a small calibration tell (session-1
    mean PIT ≈ 0.48, slightly negative calibration intercept).
  - **Grounding correction:** in the measured backtests, bootstrap ≈ Gaussian (p=0.74,
    indistinguishable) — the empirical full-distribution method did NOT beat a mean+variance Gaussian.
    The only accepted lift was **event conditioning** (+0.032, p=0.027). So "better than just the mean"
    holds only vs a degenerate point forecast; the value added so far is **dispersion**, not the mean
    and not yet the tails. (`docs/results.md`.)
  - **Expectation if we test mean-reset (shrink the drift toward the martingale):** likely a small
    log-score LIFT, growing at longer horizons (`μ·h` grows linearly, `σ·√h` as √h — noisy drift
    contaminates longer-dated forecasts more). This is a bias–variance/shrinkage argument: the
    trailing sample mean is unbiased but high-variance for a near-zero true drift; shrinking to zero
    cuts centering MSE out-of-sample. Generalize as a shrinkage intensity κ∈[0,1] tuned by log-score.
  - **TESTED (Session 11) — the expectation was WRONG in aggregate.** `DriftDampenedBootstrapForecaster`
    (terminal mean linearly shrunk toward spot by κ) on the standardized set: best κ gave +0.0009 (h=5,
    p=0.57) and +0.0008 (h=21, p=0.63) — **indistinguishable at every κ, and the predicted longer-horizon
    lift did not appear.** Root cause: the per-ticker effect is large but ~symmetric and cancels — κ
    helps names whose drift was noise, hurts names whose drift was signal. A *uniform* shrinkage can't
    separate signal from noise, so it nets to zero. This sharpens the conclusion below: the directional
    drift is neither pure noise nor pure signal but a per-name mix, so the only path to a trustworthy
    directional edge is a **conditional/informed** anchor (fundamentals), not a global reset. See
    `docs/results.md` + `docs/decisions.md`.
  - **The load-bearing tension:** the drift we'd reset is the ENTIRE source of the current directional
    BUY/SELL sweep. **If mean-reset improves the forecast score, that is direct evidence the directional
    signals are noise, not edge** — a calibrated (centered) forecaster emits ~no directional signal and
    the only honest edge left is vol/shape (our tails vs the market's IV smile). Calibration and the
    current directional edge pull on the same knob in opposite directions. A trustworthy directional
    edge must therefore come from a real signal (the fundamentals anchor above), proven by backtest —
    not from the trailing sample mean.

---

## ISSUE-N — Option-Implied (own-options) forecaster is circular; benchmark only
- **Status:** by-design constraint, not a bug. Built Session 17.
- **Concern:** `OptionImpliedForecaster` reconstructs a ticker's OWN risk-neutral PDF from its own option
  chain. Using it to *value* those same options is circular — fair value collapses onto the market price,
  so it can never produce a tradeable edge and must never be the production default.
- **Honest use:** it is a *measurement* tool — the head-to-head accuracy benchmark for whether our
  production forecast beats the market's own implied view (see `docs/decisions.md` "Option-Implied
  (own-options)" ADR and `scripts/run_option_implied_backtest.py`). Negative (option_implied −
  event_bootstrap) log-score diff ⇒ our forecast is more accurate than the prices for sale.
