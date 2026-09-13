# Architecture Decisions

Locked decisions and their rationale, deduped across sessions. Newest first.

## Ensemble: 50/50 PATH-LEVEL MIXTURE chosen over learned weights (Session 18)
- **Decision:** the first ensemble combiner is a fixed **50/50 probability mixture** of the
  event-bootstrap and event-OIB terminal price distributions (`BlendForecaster` /
  `blend_price_distributions` / `BlendFactory`; backtest `rolling_log_score_backtest_blend_event`).
  It is the to-do "Ensemble / router" approach **(d) path-level mixture**: concatenate both
  forecasters' `PriceDistribution` samples and halve each side's weights so the mixture still sums to 1.
- **Why a mixture, not an average:** averaging means/parameters would blur each forecaster's shape; a
  mixture keeps both shapes intact (fat tails, skew) — half the probability mass is each view. KDE
  log-score scoring works unchanged on the weighted mixture.
- **Why fixed 50/50, not learned/adaptive weights (the substantive reason):** with limited rolling
  backtest data, fitting blend weights (softmax of recent per-arm log scores, per-ticker selectors, etc.)
  over-uses the data and risks overfitting the *combiner* on the same window we evaluate on. Equal
  weight is the honest, no-free-parameter starting point; a learned router is a separate, held-out-gated
  experiment (still on the to-do board). Per the user: do NOT add a more complicated blend scheme now.
- **Status:** **PROMOTED to the production default (S18, per user direction).**
  `run_daily._default_forecaster_factory` now returns `BlendFactory([EventBootstrapFactory,
  OptionImpliedBetaFactory(event_source=...)], (0.5, 0.5))` — i.e. event-bootstrap ⊕ event-conditioned
  OIB. The blend cleared the criterion vs event-bootstrap on the S18 window (+0.0107, p≈6e-14, 27/40)
  but the edge is small, OIB-inherited, and single-window — so this is a judgment call to ship the more
  robust expression of the signal, NOT a strong-evidence flip. Live-verified: `run_daily --tickers AAPL`
  dry-run forecasts + sizes with no error. Regression test updated
  (`test_default_forecaster_factory_is_blend`). To make the LIVE OIB arm match the backtested event-OIB,
  `OptionImpliedBetaFactory` gained an optional `event_source` (earnings-conditioned idiosyncratic
  residuals; reuses `_conditioned_residuals` via lazy import). Each arm still degrades to bootstrap on
  feed failure. **Open risk:** single-window edge; the held-out 2nd window + vol/beta router (to-do)
  remain the proper gate. `--no-events` → plain bootstrap; `--garch-fhs` → GARCH-FHS still available.

## Forecaster: Option-Implied (own-options) — BENCHMARK ONLY, NEVER a production default (Session 17)
- **Decision:** `OptionImpliedForecaster` (`forecast/option_implied_forecaster.py`) — which extracts a
  ticker's OWN Breeden-Litzenberger risk-neutral PDF and re-anchors it to spot — is a measurement
  benchmark, NOT a tradeable forecaster. It is exposed via `OptionImpliedFactory` (live) and
  `backtest/option_implied_backtest.py` (rolling backtest), but `run_daily._default_forecaster_factory`
  stays `EventBootstrapFactory` (unchanged).
- **Why it can never be a production default (the substantive reason):** by construction it reproduces
  the very option prices that are currently for sale. Valuing those same options against it is circular —
  fair value ≈ market price ⇒ ~zero edge by definition. So it cannot generate trades; it can only
  *measure* whether another forecaster beats the market's own implied view.
- **What it IS for:** the accuracy/profitability check the user asked for. Run it head-to-head against the
  production forecaster (event-bootstrap) on the rolling log-score backtest
  (`scripts/run_option_implied_backtest.py`). The sign of (option_implied − event_bootstrap) is the
  signal: NEGATIVE ⇒ our production forecast scores higher than the market's own price ⇒ we are more
  accurate than the prices for sale ⇒ predictive edge / a basis to be profitable. ~0 or positive ⇒ the
  market price already encodes our information.
- **Contrast with Option-Implied BETA** (`option_implied_beta_forecaster.py`): that one maps the *index*
  options' PDF onto a ticker via Beta + idiosyncratic residuals, so it is NOT circular and remains a
  genuine forecaster candidate (still gated by the log-score criterion). This one uses the ticker's OWN
  options directly, which is exactly why it is only ever a benchmark.
- **Reuse:** `build_eod_pdf` and `live_eod_pdf` are symbol-generic by design (renamed from
  `build_eod_index_pdf` / `live_index_pdf` in the S18 cleanup — they always worked for any symbol,
  the old names just implied index-only), so the own-options path is almost entirely existing machinery.

## Forecaster: GARCH-FHS ACCEPTED (plain, ungated); vol-divergence gate REJECTED (Session 11)
- **Decision:** adopt `GarchFhsForecaster` (GARCH(1,1)-filtered Filtered Historical Simulation) as a
  backtest-validated forecaster, in its PLAIN always-on form. Do NOT gate it on the conditional/
  unconditional vol ratio.
- **Evidence:** standardized set, h=21 (the production horizon). 40-tkr: +0.1373 vs bootstrap,
  t=3.74, p=0.0002, 57% win — the largest log-score lift of any forecaster idea to date, best of all
  four forecasters. A τ-sweep of the vol gate (`VolGatedForecaster`, use FHS when σ_{T+1}/σ̄ ≥ τ) was
  monotonically harmful: τ=0 (always FHS) −5.037 was the best arm; every gate scored worse, several
  worse than pure bootstrap. See `docs/results.md` for both tables.
- **Why ungated (the substantive reason):** FHS adds value across the WHOLE conditional-vol profile —
  it correctly widens before turbulence AND narrows on calm days (where the bootstrap over-disperses).
  The biggest gate-induced loss came from switching the calmest days OFF FHS, i.e. those were FHS wins.
  Any vol-state threshold discards genuine wins on one side. The improvement is convex/concentrated
  (huge wins on vol-regime/fat-tail names — QCOM, FTNT, SBAC; slight drag on a few calm names), but the
  mean is the right objective for option-EV/Kelly and a convex profile is desirable. The calm-NAME drag
  is a per-name fit issue, not a per-day vol-state one, so a day-level gate cannot address it.
- **Caveat held honestly:** the pooled per-forecast test (the project's standard, p=0.0002) is stronger
  than the conservative per-ticker test (n=40, p=0.28) because the win is concentrated in a few names'
  many holdout days. Accepted on the pooled test + positive median per-ticker diff (+0.030) + 57% win +
  ex-FTNT-still-positive (QCOM > FTNT, so not a one-name artifact). A per-NAME gate (persistence/ARCH-LM)
  is a possible future refinement, not pursued (plain FHS already wins and is simplest).
- **Open:** wiring plain FHS into the LIVE path (`factories.py`) is a separate deployment decision; and
  composing it with event conditioning (event-conditioned FHS) is a natural follow-up.

## Forecaster: drift-dampening (uniform mean-reset toward spot) — REJECTED as a default (Session 11)
- **Decision:** do NOT make `DriftDampenedBootstrapForecaster` (terminal mean linearly shrunk toward
  spot by a global κ) a default. It does NOT clear the log-score decision criterion. The forecaster is
  kept in the tree (reachable via `scripts/run_standardized_test.py --drift-dampening-experiment`) as a
  measurement tool, not a production forecaster.
- **Evidence:** standardized 20-ticker set. h=5: best κ=0.25 = +0.0009 (t=0.57, p=0.57, 45% win).
  h=21: best κ=0.10 = +0.0008 (t=0.49, p=0.63, 50% win). Indistinguishable at every κ; full reset
  (κ=1.0) is strictly worse. See `docs/results.md` for the per-ticker table.
- **Why (the substantive finding, not just "p>0.05"):** the per-ticker effect is large but roughly
  symmetric and cancels in the pool — κ helps names whose recent drift was noise and hurts names whose
  recent drift was signal. A single global shrinkage knob can't tell the two apart, so it can't add
  value on average. This is the empirical confirmation of the ISSUE-1 tension: trailing drift is a
  *mix*, and the fix is a **conditional / informed** fair-value estimate (the fundamentals anchor), not
  a uniform reset. The longer-horizon lift that shrinkage theory predicted (μ·h grows vs σ·√h) did not
  appear — h=21 was no better than h=5.
- **Mechanism note:** the experiment used a pure ADDITIVE translation of terminal prices (per the
  request), which preserves dispersion/shape and isolates the drift effect. A log-space drift shrink
  (rescaling per-day μ, which also rescales dispersion) is a *different*, untested experiment — but
  given the cancellation result above, a uniform version of it is unlikely to fare differently.
- **Like the block-bootstrap and cheap-option-cap rejections, recorded so it isn't re-proposed.**

## Sizing: cheap-option contract cap — REJECTED (Session 10)
- **Decision:** do NOT add a cap that limits contract count / suppresses cheap deep-OTM options
  (was on the backlog as a "per-contract/notional sanity cap"). (User-chosen.)
- **Rationale (why the cap is wrong, not just deferred):**
  - **Tails are the edge, not the weak spot.** The whole thesis is that our forecast differs from the
    market's ~normal/IV pricing *in the tails/shape* — exactly what `decompose_edge` isolates as the
    "shape edge" (the part we trust *more* than drift / ISSUE-1). A cap whose effect is "smaller bets on
    OTM options" suppresses the precise disagreement the model exists to express. The earlier "tail =
    least reliable part of the model" framing wrongly treated *signal* as *noise*.
  - **Concentration is already bounded** by the per-name (5%) + gross (20%) caps and diversification
    across the 40-name watchlist. A high *contract count* on a cheap option is just dollar-budget ÷
    cheap premium — ~$1,800 / ~2% of bankroll for the HBAN 100× example — not extra risk.
  - **Liquidity is handled by limit orders** — we post a limit, so the downside is a no-/partial fill,
    never a bad fill. No slippage-through-our-price risk.
  - **The one real residual — Kelly oversizing on noisy tail probabilities — is already handled by the
    right lever:** fractional Kelly (¼) + the max-fraction cap ARE the uniform haircut for input
    estimation error. A cheap-option special case would instead bake in a structural anti-tail bias.
    If a tail bet feels too aggressive, tune the Kelly fraction, not a carve-out.
- **Thesis-aligned alternative:** improve the tail itself (fat-tail Student-t / jump-diffusion
  forecaster, top of the forecaster backlog), gated by the log-score backtest — the opposite of a cap.
- **Like the block-bootstrap rejection, recorded so it isn't re-proposed.**

## Portfolio construction (Session 6)
- **Gross premium cap = total capital-at-risk cap.** For LONG options premium = max loss, so capping
  total premium at `max_gross_fraction` (default 20%) of bankroll is a precise worst-case-loss bound.
- **Greedy-by-log-growth allocation** (not pro-rata) when candidates exceed the budget — fully fund
  the best signals top-down, skip the rest. Kelly value is convex in edge quality; diluting strong
  edges to fund marginal ones is anti-Kelly. (User-chosen.)
- **Unhedged-book overall haircut, NOT a hard directional cap (Session 7, supersedes the S6 cap).**
  The earlier hard cap (`max_direction_share`, reject any candidate past 60% of one side) was dropped:
  it refused signals the forecaster genuinely saw and left budget idle. Now we ALLOW a one-sided book
  but PENALISE it — if the funded book is unhedged (one side > `hedge_threshold`, default **80%**, of
  deployed premium) the whole gross budget is cut by `unhedged_haircut` (default **20%**) and the book
  is re-allocated. Implementation is two-pass: allocate at full budget to discover the natural book,
  test hedge, shrink + re-allocate if unhedged. (User-chosen: "drop direction_cap; reduce the overall
  bet 20% if we fail to hedge.") Note: the haircut bites only when the book is large enough to be
  budget-constrained; a small one-sided book under budget is unaffected. Joint/correlation-aware Kelly
  remains a future, backtest-gated upgrade.
- **Existing open-option premium is charged against the budget** so re-runs don't stack exposure.
- **Portfolio caps are a risk-policy choice, not a forecaster change** → NOT gated by the log-score
  criterion. A future joint-Kelly upgrade would be judged on portfolio P&L / drawdown, not log score.

## Valuation (Session 5)
- **`fair_value` is a P-measure EV, NOT a no-arbitrage price.** The valuer discounts
  `E_P[max(S_T−K,0)]` under the forecast's real-world distribution. The market quote and BS price
  the option under the risk-neutral (Q) measure. The edge is exactly that P-vs-Q gap — the whole
  thesis. Conflating the two is the project's central conceptual trap, called out in both modules.
- **Held-to-expiry only (V1).** The `PriceDistribution` must be forecast at the horizon equal to the
  contract's expiry; terminal-payoff valuation is then exact and needs no price path. Caller owns
  that consistency. Early-exit valuation (needs IV-at-exit / path) is deferred. (User-chosen fork.)
- **European exercise assumed.** Exact for calls on non-dividend payers (early exercise never
  optimal); a small approximation with dividends.
- **Per-share valuation**, matching how bid/ask are quoted. The 100x contract multiplier is a
  dollar-sizing concern, deferred to the Kelly layer — it doesn't affect edge.
- **Edge measured against the side actually transacted**: buy clears the ask, sell must beat the
  bid. EV inside the spread ⇒ HOLD (no edge after the half-spread). No-quote ⇒ NO_QUOTE.
- **Black-Scholes kept only as a reference** (sanity-check the EV machinery against a closed form;
  back out market implied vol for comparison) — not on the pricing path.

## Standing: forecaster acceptance = log-score lift in backtest
A forecaster change ships only if it raises mean log score in the rolling OOS backtest, paired
vs the incumbent across the 20-ticker standardized set. Not intuition / in-sample fit / theory.
(Also in CLAUDE.md and the persistent memory.)

## Events
- **EventSource = Composite** (yfinance earnings + manual macro/FDA CSV; `AlpacaNewsSource` later)
  behind a swappable Protocol. yfinance gives a *structured* earnings calendar (historical +
  upcoming) — better than Alpaca News (unstructured) for clean earnings identification.
- **`ReturnDistribution` stays PURE** — event labels are NOT folded into it (rejected "Option B").
  Decisive reason: Option B cannot represent the *forward* horizon schedule (future dates have no
  historical samples to label); event knowledge must live in the calendar/forecaster layer anyway.
  Same single-responsibility logic as keeping sampling strategy out of `ReturnDistribution`.
- **`ConditionedReturnDistributions` is the bridge** (composite of pure `ReturnDistribution`s keyed
  by type) — lives in `forecast/`, composes but never mutates `ReturnDistribution`.
- **Event forecaster binds schedule + distributions at construction**, leaving the shared
  `forecast(horizon_days, spot)` signature unchanged.
- **timing (BMO/AMC) is first-class** on `Event` — it determines the return-index attribution.
  UNKNOWN → treated as AMC (large-cap norm).
- **Per-ticker only event samples** for V1 (~11 earnings/3y; thin, accepted). Cross-ticker pooling
  is a backlog item.
- **No price look-ahead in the conditioned backtest**, but KNOWN future event *dates* are used
  (legitimate — announced in advance; the label carries no price info). Verified empirically.

## ReturnDistribution / forecasters
- **`ReturnDistribution` is a pure data container**; sampling strategy lives in the `Forecaster`
  subclasses (Strategy pattern). A `draw()` primitive on `ReturnDistribution` was discussed as the
  one defensible shared helper, but not adopted (kept the smell small rather than refactor).
- **`spot` is a forecast-time arg, `n_paths` is constructor-time** — lets one forecaster be
  re-queried at different spots.
- **`GaussianForecaster` uses the weighted moments of whatever distribution it's given** — so a
  head-to-head vs bootstrap isolates the parametric-vs-empirical question.
- **Exponential decay parametrized by λ∈(0,1) directly** (default 0.99, ~69-day half-life), not
  half-life days. Silverman auto-bandwidth for KDE smoothing.
- **`add_sample` maintains unnormalized `_raw_weights`** so incremental update == rebuild (naive
  decay-then-renormalize over-weights the new sample).

## Block bootstrap
- **Overlapping (moving-block) windows** over non-overlapping (sample-starved); shared-day
  correlation is harmless when sampling, not doing inference.
- **End-anchored block weighting** (newest day's decay weight) — recency-consistent.
- **Reconstruct blocks from the daily `ReturnDistribution`** (samples already time-ordered) rather
  than a separate `KDayReturnDistribution` — no new container or harness changes.

## Standardized test / universe
- **S&P 500** as the cap-weighted universe (~80% US equity cap; small-cap tail negligible under
  cap weighting).
- **Frozen committed snapshot, not live caps** — required for reproducibility; refresh is deliberate.
- **Per-iteration seed bump + common random numbers across forecasters** — decorrelates
  within-forecaster MC noise while keeping the paired diff low-variance.
- **Report filenames stamped `_h<H>_hold<HO>`** — so runs at different settings don't clobber.
- Known biases documented: survivorship (today's constituents), GOOGL/GOOG dual-class.

## Data
- **Module-level constants over runtime introspection** for validation enumeration (`OHLCV_FIELDS`).
  (User preference.)
- 3-year default lookback (matches the JD calibration window in the original plan).
