# Results

Forecaster evaluation history. Metric = mean **log score** (higher is better) in the rolling OOS
backtest. Standardized set = META + 19 cap-weighted S&P 500 tickers @ seed 42 (`sp500_2026-06-02`).

## Leaderboard — standardized set, h=5, holdout=63, n_paths=10k, seed=42 (n=1180 each)
Event source = earnings (yfinance) + **FOMC (auto-fetched)** + manual, as of Session 9.
| Forecaster | mean log score | vs bootstrap | verdict |
|---|---|---|---|
| **event_bootstrap** | **−4.050** | +0.039, t=2.54, **p=0.011** | ✅ accepted (best) |
| bootstrap (default) | −4.089 | — | incumbent |
| gaussian | −4.134 | −0.045 | baseline |
| block_bootstrap | −4.159 | −0.070 | ❌ rejected |

Per-ticker win rate event vs bootstrap: **55% (11/20)**. Lift concentrated in a few big wins
(FTNT −3.79→−3.29, WMT −3.08→−2.80, UNP, LLY, TMO, GOOGL); CDNS/BSX/WM/PLTR slightly worse. Only a
minority of the 59 holdout windows per ticker straddle an event, so the per-window effect is larger
than the pooled diff. Artifacts: `output/standardized_test_2026-06-05_h5_hold63.{md,csv}`.

### FOMC added to the event source (Session 9)
Re-ran `--events` after wiring `FomcCalendarSource` into the event composite. event_bootstrap vs
bootstrap went from **+0.032 (p=0.027, 50% win)** [earnings + stale manual FOMC] → **+0.039 (p=0.011,
55% win)** [earnings + auto-fetched FOMC]. Adding Fed days **did not degrade and plausibly improved**
the lift, and it stays accepted by the decision criterion. **Caveat:** this is not a clean FOMC
ablation — it changes the event SET (also gives FOMC fuller history + BMO/announcement-day timing),
so it does not isolate FOMC's marginal contribution. A rigorous earnings-only vs earnings+FOMC
head-to-head (and a BMO-vs-AMC timing comparison) is the follow-up before attributing the gain to FOMC
specifically. The earlier earnings-only numbers are preserved in this file's history below.

#### Prior earnings-only event result (superseded as the live default, kept for the record)
event_bootstrap −4.034, +0.032 vs bootstrap, t=2.21, **p=0.027**, 50% win (10/20). Source = earnings +
manual (stale/past FOMC, AMC-tagged). Artifacts: `output/standardized_test_2026-06-02_h5_hold63.{md,csv}`.

## Block bootstrap at h=10 (loses by more)
bootstrap −4.527 > gaussian −4.584 > block −4.764 (n=1080). Block vs bootstrap: **−0.238,
t=−5.17, p=2.8e-07**, win rate 20%. Longer horizon did NOT rescue block (overlapping 10-day
windows share 9/10 days → fewer independent shapes; iid diversity compounds). Artifacts:
`..._h10_hold63.{md,csv}`.

## 50/50 blend (event-bootstrap ⊕ event-OIB) — Session 18
Path-level 50/50 probability mixture (`BlendForecaster`); `scripts/run_blend_backtest.py`. Window
2022-06→2025-06, h=21, holdout=252 (eval ≈ 2024-05→2025-05). OIB arm = event-conditioned.

**Item 1 + 3a — blend vs event-bootstrap (40 names, n=7880, seed 11):**
| arm | mean log score | vs event_bootstrap |
|---|---|---|
| event_bootstrap (production) | −3.9803 | — |
| **blend 50/50** | **−3.9696** | **+0.0107, t=7.5, p≈6e-14, per-eval win 51%, 27/40 tkrs** ✅ |
| oib (event) | −3.9677 | +0.0126, t=4.7, p≈2e-6, 26/40 tkrs |
- blend − oib = −0.0019 (p=0.23, NOT significant): the blend's gain over event-bootstrap is ~85%
  inherited from OIB; it does not beat OIB. Blend sits BETWEEN the two arms on 33/40 names (hedge),
  beats both on only 6 → its value is **robustness/lower per-ticker variance**, not outright gain.
- Edge is **tail-driven, not frequent**: per-eval OIB-vs-eboot median is −0.002 (win ~49%) but mean
  +0.013 — OIB wins bigger when it wins. Same for the blend (median ≈0, mean +0.011).
- **Cross-sectional (item 1):** OIB beats event-bootstrap on high-vol/cyclical/growth names
  (APH +0.102, CEG +0.101, IEX +0.086, ROST +0.056, AMD/AMZN +0.050, COHR, HOOD) and loses on
  defensives/healthcare (UNH −0.066, SBUX −0.056, CBOE −0.039, DGX, SBAC, ABT) — matches the
  documented "OIB→high-beta, bootstrap→low-vol defensive" split. Chart: `output/blend_compare.png`.
- **Caveat:** ONE window. OIB-vs-event-bootstrap has been regime-dependent (S16: W1 +0.12 / W2 ≈0), so
  the durable claim is "blend tracks the better arm with lower variance," not a large absolute edge.

**Item 3b — blend vs the ticker's OWN option price (option_implied) (20→19 names, n=1728, seed 23):**
- Pooled: blend −4.060 vs option_implied −6.222 → **+2.16, t=14.9, p≈4e-47, 19/19 tkrs**. ✅ but
  **HEAVILY inflated by thin chains** — illiquid names' own-options PDF is badly estimated and gets
  annihilated on log score when it assigns ~0 density to the realized move (WDC +10.6, HPE +5.9,
  CVX +7.0, PSA +5.7, n=4–55).
- **Honest liquid-only read:** restricting to the 5 names with n≥150 (META, MSFT, PYPL, NKE, WFC) →
  mean **+0.729 but MEDIAN −0.026, win 47%**. Mega-caps META/MSFT only **+0.06**. So the blend beats the
  market-implied PDF **in expectation (tails)**, but on a TYPICAL day the liquid option price is as
  accurate or marginally better (median<0, win<50%) — a fat-tail/calibration edge, consistent with the
  S17 own-options quantile finding. Chart: `output/blend_vs_options.png`.

**Decision:** blend clears the criterion vs event-bootstrap on this window (+0.0107, p≈6e-14); gain is
small, inherited from OIB, and single-window (OIB's edge is regime-dependent). **Default FLIPPED to the
blend (S18, per user direction)** — `_default_forecaster_factory` now returns `BlendFactory(
[EventBootstrapFactory, OptionImpliedBetaFactory(event_source=...)], 0.5/0.5)`; live dry-run verified.
Shipped as the more robust *expression* of the OIB signal, not on strong evidence — the held-out 2nd
window and/or vol/beta router (to-do) remain the proper gate. (50/50 over learned weights to avoid
overfitting the combiner — `docs/decisions.md`.)

## Option-Implied (own-options) BENCHMARK vs production (Session 17) — accuracy vs the market price
A diagnostic, not a forecaster gate. `OptionImpliedForecaster` reconstructs a ticker's OWN
risk-neutral PDF (the prices for sale) and is scored head-to-head against the production default
(event-bootstrap). The sign of (option_implied − event_bootstrap) is the read: NEGATIVE ⇒ our forecast
is more accurate than the market's own implied view (predictive edge / a basis to profit); ~0/positive
⇒ the market price already encodes our information. (It is BENCHMARK-ONLY — circular to trade against;
see `docs/decisions.md`.)

**Live smoke (illustrative, NOT a gated claim):** META, 2024-06→2025-03, h=21, holdout=30, n_paths=4k,
n=10 paired, 0 skipped. option_implied −5.688 vs event_bootstrap −5.453 → **diff −0.235, t=−2.0,
p=0.075, win 40%** — directionally "we beat the market price" on this tiny window, but n=10/one ticker
is far too small to claim edge. Purpose here was to confirm the pipeline runs end to end vs Alpaca
historical chains. Run `scripts/run_option_implied_backtest.py --sample N` over many tickers for a real
read. Artifact: `output/option_implied_backtest_smoke.md`.

### WHERE we beat the market price — by realized-return quantile (Session 17, 30 tkr, 1.5yr)
`scripts/analyze_option_implied_quantiles.py --sample 30 --sample-seed 7 --power 0.5 --start 2022-06-01
--end 2026-05-01 --horizon 21 --holdout 378 --n-paths 4000`. **n=4530 paired evals, 27/30 tickers**
(TECH + SNA contributed 0 — no liquid historical chain; coverage is liquidity-weighted, dominated by
large-caps: GOOG 358, ORCL 340, NVDA 331, WFC 324, F 319). advantage = log-score(event_bootstrap) −
log-score(option_implied); >0 ⇒ our forecast more accurate than the market's own price.

**Overall: mean advantage +1.52 but win rate only 46.3% (<50%)** — the positive mean is a pure
FAT-TAIL effect, not a broad edge. The edge is **U-shaped in realized return**: strong at both extremes,
negative-median through the bulk. Median is the robust stat (mean is dominated by a handful of tail
evals where the near-lognormal market PDF assigns ~0 density to a big move and gets crushed on log score).

| return decile | range | mean adv | **median adv** | **win%** |
|---|---|---|---|---|
| 1 (big down) | −36%..−11% | +4.02 | **+0.51** | 61% |
| 2 | −11%..−6.8% | +1.43 | −0.29 | 34% |
| 3 (worst) | −6.8%..−3.6% | +0.86 | −0.38 | 19% |
| 4 | −3.6%..−1.0% | +0.54 | −0.27 | 22% |
| 5 | −1%..+1.2% | +0.62 | −0.13 | 37% |
| 6 | +1.2%..+3.2% | +0.58 | −0.03 | 48% |
| 7 | +3.2%..+5.5% | +0.86 | +0.05 | 55% |
| 8 | +5.5%..+8.1% | +0.56 | +0.07 | 56% |
| 9 | +8.1%..+12% | +0.60 | +0.03 | 53% |
| **10 (big up, BEST)** | **+12%..+53%** | **+5.12** | **+1.32** | **80%** |

**Best quantile = the top return decile (large up-moves, +12%..+53%): median +1.32, 80% win.** Bottom
decile (large drops) is runner-up (median +0.51, 61%). Through deciles 2–5 (everyday small moves) the
median advantage is NEGATIVE — the market's smooth implied PDF is better calibrated there; our worst
zone is small down-moves (decile 3, 19% win). **Interpretation:** our event-conditioned bootstrap keeps
real earnings-jump shapes + fatter tails, so it dominates when the move is extreme but loses in the
bulk. For option EV (convex, tail-driven) this is the useful kind of edge, but it is concentrated and
noisy — consistent with ISSUE-1 (forecaster expresses tail/shape, not reliable everyday direction).
**Caveat:** decile bins are by realized return (an OUTCOME) — this maps WHERE we're accurate, it is NOT
a tradeable signal. Artifacts: `output/option_implied_quantiles.{png,csv,md}`.

## Option-Implied Beta — TWO-WINDOW verdict (Session 16 cont.): edge is a REGIME ARTIFACT, not robust
The single-window 100-ticker run below looked decisive (+0.046 vs bootstrap, p≈1e-12). A second,
non-overlapping window overturns it. Plain three-way (oib / bootstrap / plain GARCH-FHS), h=21,
holdout=63, n_paths=10k:

| Window (eval dates) | tkrs | oib−boot | p | oib−garch | garch−boot (sanity) |
|---|---|---|---|---|---|
| W1 — Feb–Apr 2025 (tariff-vol stress) | 100 | **+0.046** | 1.6e-12 | +0.103 | **−0.058** (wrong sign) |
| W2 — mid-2024 (calm) | 198 | **−0.005** | 0.31 | −0.032 | **+0.026** (right sign) |

**Read:** the garch−bootstrap sanity check flips sign across windows — negative in W1, positive (≈its
true +0.137 direction) in W2 — proving W1 is an unusual regime, not a harness issue. OIB's edge tracks
exactly that: a huge lift in the W1 vol stress (forward-looking option vol shines), **gone in calm W2**
(−0.005, p=0.31, 85/198 ≈ coin flip; and −0.032 behind GARCH-FHS). So OIB is a **vol-stress-conditional**
forecaster, not a robust unconditional one. **Verdict: does NOT clear the decision criterion as an
unconditional default.** Artifacts: `output/oib_3way_100tkr.{md,log}` (W1), `output/oib_3way_200tkr_w2.{md,log}` (W2).
A possible future framing: deploy OIB only when conditional/implied vol is elevated (a regime gate) —
untested. Event-conditioned three-way (vs the real default, event-GARCH-FHS) on both windows: pending
(`--events`, 60 tkr).

### Event-conditioned two-window three-way (the fair fight vs the REAL default)
All three arms event-aware: **event-bootstrap**, **event-GARCH-FHS** (the production default), and
**event-OIB** (idiosyncratic residuals conditioned on earnings; market events stay free via the index
PDF). 60 tkrs (seed 42, power 0.5), h=21, holdout=63, n_paths=10k.

| Window | oib−event_boot | oib−event_garch | event_garch−event_boot (sanity, exp ~+0.224) |
|---|---|---|---|
| W1 (Feb–Apr 2025 stress) | **+0.120** (p=6e-29) | **+0.048** (p=0.003, 33/60) | +0.072 (p=5e-4) |
| W2 (calm mid-2024) | −0.001 (p=0.86) | **+0.026** (p=6e-4, 34/59) | −0.027 (p=3e-5) |

**Read (nuanced):**
- **vs the simple event-bootstrap baseline:** same regime story as plain — OIB crushes it in the W1
  vol stress (+0.120) but ties in calm W2 (−0.001). NOT a robust edge over the baseline.
- **vs the event-GARCH-FHS production default:** OIB beats it in BOTH windows (+0.048, +0.026, both
  significant, ~55–58% of tickers) — the only comparison that's positive in both regimes. BUT this is
  a **soft win**: event-GARCH-FHS badly underperformed its own leaderboard +0.224 here (only +0.072 in
  W1, and **−0.027 in W2** — worse than event-bootstrap), i.e. these 2024–25 windows are unfavourable to
  GARCH, so "OIB > event-GARCH" is partly "GARCH was weak here," not purely "OIB was strong."

**Overall verdict:** borderline-positive vs the real incumbent but NOT a clean robust win — OIB can't
robustly beat the *simple* event-bootstrap (W2 tie), and its win over event-GARCH leans on GARCH's
anomalous weakness in these windows. **Not flipping the default on this evidence.** Two robust, useful
truths emerge: (1) OIB's large edge is **vol-stress-conditional** (a regime gate is the live angle worth
testing), and (2) OIB is **never much worse than the baseline** and has big upside in stress — a decent
hedge/ensemble member. Clean tie-breaker: a 3rd window where event-GARCH recovers ≈+0.224 (trustworthy
incumbent), check OIB still edges it. Artifacts: `output/oib_3way_event_w{1,2}.md`, `oib_3way_event_both.log`.

### Where each forecaster dominates (per-ticker, ensemble-relevant) — Session 16 cont.3
The pooled W2 tie HIDES a stable cross-sectional structure. Per-ticker, averaged over the two event
windows, the **best** forecaster splits **OIB 24 / event-GARCH 21 / event-bootstrap 14** of 59 — all
three own a real slice, so an ensemble/router is well-motivated. The split is largely by **volatility /
beta level**, and it is **stable across both regimes** (not a W1 artifact):

- **OIB's stable territory — HIGH-vol / HIGH-beta names** (18 beat bootstrap in BOTH windows): airlines
  UAL (+0.50/+0.15), cruise RCL, luxury RL (+0.44/+0.17), megacap growth META (+0.42/+0.16) / NVDA,
  semis AMAT/APH/APP, cyclical industrials DOV/AME/APD/JBL. Edge biggest in the W1 stress but still
  clearly positive in calm W2 for these names. corr(OIB−boot edge, vol-level proxy) = **+0.55**; high-vol
  tercile mean OIB−boot W1 **+0.29**, low-vol tercile ≈ 0.
- **Event-bootstrap's territory — LOW-vol DEFENSIVES** (12 beat OIB both windows): utilities ED/PCG,
  REIT O, insurers PGR/ALL, BRK-B, AZO, LMT, ADM. OIB's risk-neutral PDF over-disperses these calm names.
- **Event-GARCH's small niche** (beats BOTH others in both windows): C, CB, ADM, PCG — a few
  banks/insurers/utility/staples where conditional-vol modelling specifically helps.
- 27 of 59 are regime-flippers (the indeterminate middle).

**Ensemble implication:** route by vol/beta — OIB for high-vol/high-beta names, bootstrap for low-vol
defensives — captures OIB's upside without the defensive drag. The robust, ensemble-usable signal is
CROSS-SECTIONAL (which ticker), layered on a TEMPORAL one (how much, scaled by regime). Decision for now:
**keep the incumbent default; do not adopt OIB standalone.** Build the router later.

### W1 detail — 100-ticker single-window (superseded by the two-window verdict above)
Backtest of the option-implied-beta forecaster (historical SPY/IWM EOD option PDFs via
Breeden-Litzenberger → Beta map + idiosyncratic residual). Point-in-time rolling, three-way paired
on the SAME dates vs **bootstrap** (baseline) and **plain GARCH-FHS** (strongest non-event forecaster).
Driver: `scripts/run_option_implied_beta_backtest.py`.

### 100-ticker three-way (the headline)
100 cap-weighted-√ tickers (seed 42, power 0.5), h=21, holdout=63, window 2023-01-01..2025-05-01
(eval dates ≈ **Feb–Apr 2025**), n_paths=10k. 0 dates skipped, 0 ticker failures, n=4300.
Artifact: `output/oib_3way_100tkr.{md,log}`.

| Forecaster | mean log score |
|---|---|
| **option_implied_beta** | **−4.184** |
| bootstrap | −4.230 |
| garch_fhs (plain) | −4.287 |

| Pair | diff | t | p | per-ticker | verdict |
|---|---|---|---|---|---|
| oib − bootstrap | **+0.046** | 7.09 | 1.6e-12 | **77/100** | ✅ ACCEPT |
| oib − garch_fhs | **+0.103** | 9.99 | 3e-23 | 58/100 | ✅ ACCEPT |
| garch_fhs − bootstrap | **−0.058** | −5.49 | 4e-8 | 45/100 | ⚠️ **sanity FAIL** |

**Convex per-ticker pattern (coherent with the thesis):** big wins on high-vol/high-beta names
(UAL +0.481, RL +0.304, ISRG +0.277, META +0.223, AAPL +0.192, APP +0.162) where forward-looking
option vol + fat tails matter; small losses on low-vol defensives (CVX −0.091, INVH −0.089,
BRK-B −0.080, PGR/PCG/SYY ≈ −0.04) where the risk-neutral PDF over-disperses vs their calm realised
behaviour. Wins are large, losses small → 77/100 net positive.

**⚠️ Why NOT accepted as a default despite p≈1e-12.** The sanity check FAILED: plain GARCH-FHS scored
**−0.058 vs bootstrap here, the opposite of its leaderboard +0.137** (S11, h=21). Same harness, same
scoring — so the *eval window* is the difference: Feb–Apr 2025 was the tariff-vol stress, a regime that
rewards FORWARD-looking vol (OIB, which reads tomorrow's risk straight off option prices) and punishes
GARCH's BACKWARD-looking conditional vol. So this single window is unusually favourable to OIB; the
+0.046/+0.103 magnitudes are very likely **inflated by the regime**, and one window cannot establish
robustness. **Decision: do NOT flip the production default yet.** The gating next step is a **second,
non-overlapping window** (ideally one where garch−bootstrap reproduces ≈+0.137, confirming the harness
and isolating regime) — OIB must keep its edge there too. (Earlier 10-ticker single-window run, for the
record: oib−boot +0.078, p=0.17, 7/10 — superseded by the above.)

## Event conditioning vs GARCH complexity (Session 16 cont.4) — GARCH's machinery is NOT justified
Canonical standardized set (META + 19 cap-weighted, seed 42), **h=21**, holdout=63, n_paths=10k, recent
window (the regime where GARCH-FHS was validated). Driver: `scripts/run_garch_vs_bootstrap_check.py`;
artifact `output/garch_vs_eventboot_2026-06-22_h21_hold63.md`.

| Forecaster | mean log score | vs bootstrap |
|---|---|---|
| garch_fhs_events (production default) | −4.889 | +0.184 (p=0.002) ✅ |
| event_bootstrap | −4.920 | **+0.153 (p=0.005)** ✅ |
| bootstrap | −5.073 | — |

**Head-to-head: garch_fhs_events − event_bootstrap = +0.031, t=0.87, p=0.38, 60% by-ticker → NOT
significant.** So **~83% of GARCH-FHS's lift over plain bootstrap is the EVENT conditioning, not the
GARCH machinery**; the GARCH(1,1) QMLE + filtered-simulation + vol-propagation layer adds only an
insignificant +0.03 on top of the far simpler event-bootstrap. Confirms the EWMA intuition: the
decay-weighted (λ=0.99) bootstrap already tracks recent vol, so explicit conditional-vol modelling buys
little. Note event_bootstrap is much stronger at h=21 (+0.153) than at h=5 (+0.039, the old leaderboard)
— longer horizons are more likely to contain an earnings/FOMC day, so event routing matters more.
**Implication: event_bootstrap is the better DEFAULT on a complexity-adjusted basis** — nearly all the
lift, no per-iteration GARCH fit, and it makes the OIB+event-bootstrap ensemble a clean two-source
(forward-implied vs historical-empirical) combination. **DONE (S16):** switched
`_default_forecaster_factory` GarchFhsFactory → EventBootstrapFactory (GARCH-FHS still available via
`--garch-fhs-events`); regression test pins the new default.

## Conclusions
- **Serial dependence (block) does not help** unconditionally at h=5 or h=10 — kept only for the
  event-conditioned hypothesis.
- **Empirical shape ≈ Gaussian ≈ block** → the unconditional bottleneck is **fat tails** (the PIT
  U-shape), not dynamics. → fat-tail forecaster is the top open experiment.
- **Event conditioning helps** (modest, concentrated, but significant) — the first idea to beat
  plain bootstrap.

## Anchor experiment — intraday price vs stale close (Session 9)
Not a forecaster change: holds Bootstrap fixed, varies only the **spot anchor**. Models running
intraday on the day after a completed close. `stale_close` = anchor on close[t] (last completed
close); `intraday_random` = anchor on a uniform draw from the next day's [low,high]. Both forecast
h=5 to the same realised close[t+horizon]; CRN-paired (shared forecaster MC seed). Standardized set,
holdout=63, n_paths=10k (n=1180 each).

| Anchor | mean log score | vs stale_close | verdict |
|---|---|---|---|
| **intraday_random** | **−4.0138** | **+0.0751, t=4.50, p=7.6e-06** | ✅ accepted |
| stale_close (incumbent) | −4.0889 | — | incumbent |

Per-ticker win rate intraday vs stale: **90% (18/20)** — only PLTR & ACN marginally worse. Lift
(+0.075) is broad and larger than the event-conditioning lift (+0.032). **Interpretation:** a fresh
anchor sits ~1 trading day closer to the realised target than yesterday's close, removing one day of
staleness error — exactly the benefit the Session-7 live-price anchor captures in production. So this
**backtest-validates the existing live-price anchor** (it was shipped on intuition in S7; now proven).
Caveat: the lift partly reflects that the intraday anchor is a day fresher than the close grid — which
is precisely the real effect of anchoring on the live price. Artifacts:
`output/anchor_experiment_2026-06-05_anchor_h5_hold63.{md,csv}`.

## Anchor experiment — MA-blend vs fresh price (Session 11, Hypothesis 1) — ❌ REJECTED
Same harness as above; varies only the spot anchor. Candidate = smooth the fresh intraday price
toward a trailing close moving average; incumbent = `intraday_random` (the validated fresh-price
anchor). CRN-paired (each blend reuses the SAME fresh draw + a point-in-time MA over closes ≤ t).
Standardized set, h=5, holdout=63, n_paths=10k (n=1180 each).

| Anchor | mean log score | vs intraday_random | verdict |
|---|---|---|---|
| **intraday_random** (incumbent) | **−4.0346** | — | incumbent |
| ma7_blend (0.5·fresh + 0.5·MA7) | −4.1414 | −0.1068, t=−4.27, p=2.1e-05 | ❌ rejected |
| ma20_blend (0.5·fresh + 0.5·MA20) | −4.2148 | −0.180 | ❌ worse |
| ma7_only (pure MA7) | −4.3424 | −0.308 | ❌ worst |

Per-ticker win rate ma7_blend vs intraday: **20% (4/20)**. **Smoothing is monotonically harmful** —
more MA weight / longer window = lower score. **Interpretation:** the inverse of the S9 result. S9
proved fresh > stale (+0.075); an MA is a weighted step back toward staleness, so it gives back exactly
that edge. At h=5 the latest print is the best estimate of current level; smoothing adds lag, not noise
reduction. (A reversion benefit may exist at intraday/1-day horizons, not at the option-trading horizon.)
Artifacts: `output/anchor_experiment_2026-06-11_anchor-ma-blend_h5_hold63.{md,csv}`.

## Drift-dampening experiment — shrink the bootstrap terminal mean toward spot (Session 11) — ❌ REJECTED
New forecaster `DriftDampenedBootstrapForecaster`: draw like the plain bootstrap, then **linearly
translate** every terminal price so the mean becomes `(1−κ)·mean + κ·spot` (κ = `drift_dampening`;
κ=0 ≡ bootstrap, κ=1 ≡ mean reset all the way to spot). Pure additive shift → dispersion/shape
preserved exactly, only the centre moves. CRN-paired (the κ=0 arm IS the incumbent, byte-identical
draws). Standardized set, holdout=63, n_paths=10k. κ grid {0, 0.1, 0.25, 0.5, 0.75, 1.0}.

**h=5 (n=1180 each):** best κ=0.25 → −4.0286 vs incumbent −4.0295 = **+0.0009, t=0.57, p=0.572, 45% win.**
**h=21 (n=860 each):** best κ=0.10 → −5.0618 vs incumbent −5.0626 = **+0.0008, t=0.49, p=0.625, 50% win.**
Both **indistinguishable** → fails the decision criterion at every κ. Full reset (κ=1.0) is clearly
*worse* at both horizons (h5 −4.037, h21 −5.094); the pooled mean is flat for κ≤0.25 then declines.

**Interpretation (the key finding):** the per-ticker effect is large but ~symmetric and cancels in the
pool. A globally-applied κ helps names whose recent drift was noise (κ↑ improves META, TMO, WMT, WM,
SBAC, HUM, WMB, PLTR monotonically) and hurts names whose recent drift was signal (κ↑ worsens BSX,
GOOGL, FTNT, KLAC, AAPL, NVDA, AVGO, UNP, LLY). Net ≈ 0. This is direct evidence for the ISSUE-1
"load-bearing tension": trailing drift is a *mix* of signal and noise, and one global shrinkage knob
can't separate them — so it can't add value on average. The lift theory expected at longer horizons
(μ·h vs σ·√h) did NOT materialise: h=21 is no better than h=5. Forecaster kept in the tree (not a
default), reachable via `--drift-dampening-experiment`. A *conditional* drift estimate (the fundamentals
anchor, ISSUE-1) — not a uniform reset — remains the live path to a trustworthy directional view.
Artifacts: `output/dampening_experiment_2026-06-12_dampening_h{5,21}_hold63.{md,csv}`.

## GARCH-FHS + Event-conditioned (GarchFhsEventForecaster) — Session 12 — ✅ ACCEPTED; new production default
**Leaderboard — standardized set, h=21, holdout=63, n_paths=10k, seed=42 (n=860 each, 20 tickers)**
`sp500_2026-06-02`; event source = yfinance earnings + FOMC + manual CSV.

| Forecaster | mean log score | vs bootstrap | verdict |
|---|---|---|---|
| **garch_fhs_events** | **−4.868** | **+0.224, t=3.65, p=0.000276** | ✅ accepted — **new best / new default** |
| event_bootstrap | −4.892 | +0.200 | (incumbent, now superseded) |
| garch_fhs | −4.959 | +0.107 | ✅ accepted (S11) |
| bootstrap | −5.091 | — | baseline |
| gaussian | −5.051 | +0.040 | baseline |
| block_bootstrap | −5.813 | −0.722 | ❌ rejected |

Per-ticker win rate garch_fhs_events vs bootstrap: **65% (13/20)**.
garch_fhs_events vs event_bootstrap: **pooled +0.024 (13/20 win rate)**; formal paired t-stat not computed
for this head-to-head — direction is clear, significance left as open follow-up.

Artifacts: `output/standardized_test_2026-06-16_h21_hold63.{md,csv}`.

**Decision:** garch_fhs_events passes the standing criterion (beats bootstrap p=0.000276) and improves on the
prior event_bootstrap incumbent on pooled mean. `GarchFhsEventFactory` wired as the new production default
in `_default_forecaster_factory()` (replacing `EventBootstrapFactory`). `--garch-fhs-events` CLI flag kept as
an explicit opt-in alongside `--events` and `--garch-fhs`.

## GARCH-FHS — Filtered Historical Simulation (Session 11) — ✅ ACCEPTED (convex/concentrated)
New forecaster `GarchFhsForecaster`: fit GARCH(1,1) (variance-targeting Gaussian QMLE, scipy — no new
deps), filter the returns to standardized residuals `z_t = ε_t/σ_t`, then simulate forward by
bootstrapping `z` and propagating the variance recursion (anchored on today's `σ_{T+1}`). Fixes the iid
bootstrap's constant-vol blind spot (PIT U-shape) by making the forecast width CONDITIONAL on today's
vol state. Uniform constructor → registered in `FORECASTER_CLASSES`; CLI `--garch-fhs`.

| Set / horizon | garch_fhs | bootstrap | pooled diff | pooled t,p | win rate |
|---|---|---|---|---|---|
| 20-tkr, h=5  | −4.0028 | −4.0295 | +0.0267 | t=1.10, p=0.27 | 45% (9/20) |
| 20-tkr, h=21 | −4.956  | −5.063  | +0.1066 | t=2.13, p=0.034 | 55% (11/20) |
| **40-tkr, h=21** | **−5.0372** | **−5.1745** | **+0.1373** | **t=3.74, p=0.0002** | **57% (23/40)** |

Best of all four forecasters at every setting; the **largest lift of any forecaster idea to date**
(> event-conditioning +0.039, > intraday-anchor +0.075), at the production horizon (~21 DTE).

**Robustness (40-tkr, h=21) — the improvement is CONVEX, not broad.** Median per-ticker diff +0.030
(slim majority benefit); biggest wins are vol-regime / fat-tail names — QCOM +3.69, FTNT +2.79, SBAC
+0.75, AMZN +0.30, BX +0.30; the losers are CALM/steady names — BSX −0.85, AMD −0.82, HUM −0.76, TXN
−0.46, MU −0.40. Ex-FTNT still +0.069 (56% win), so it is NOT a single-name artifact (QCOM > FTNT, and
multiple independent winners). But drop the top-3 winners → −0.047, and the conservative per-TICKER
t-test (n=40) is t=1.10, p=0.276. The pooled per-forecast test (the project's standard, how event
conditioning + the anchor were accepted) credits the big winners' many holdout days → p=0.0002.
**Verdict:** passes the standing criterion decisively on the pooled test; the right objective for this
system (forecast → option EV → Kelly) is the MEAN, and a convex profile (dramatically better on the
big-move days options pay off on) is desirable. Caveat held honestly: the win is concentrated in
tail/vol-regime names; calm names get a slight drag. → motivates a vol-divergence GATE (below).
Artifacts: `output/standardized_test_2026-06-12_h{5,21}_hold63.{md,csv}` (the h21 csv = the 40-name run).

### Vol-divergence gate (Session 11) — ❌ REJECTED; ship plain FHS
`VolGatedForecaster`: use FHS when `σ_{T+1}/σ̄ ≥ τ`, else bootstrap. Swept τ on the 40-name set, h=21.
**Gating is monotonically harmful** — plain always-on FHS (τ=0) is the best arm and every gate is worse:

| τ | 0.00 (always FHS) | 0.90 | 1.00 | 1.10 | 1.25 | 1.50 | ∞ (always boot) |
|---|---|---|---|---|---|---|---|
| pooled log score | **−5.0372** | −5.132 | −5.133 | −5.192 | −5.184 | −5.186 | −5.1745 |

**Why it backfired (informative):** the largest single drop is the FIRST step (τ=0→0.90), which switches
only the CALMEST days (ratio<0.9) from FHS to bootstrap — so FHS was WINNING on calm days, by correctly
*narrowing* the forecast when conditional vol is below baseline (the bootstrap over-disperses there). The
edge is the whole conditional-vol profile, not just widening before turbulence; any threshold throws away
wins on one side. The earlier per-ticker calm-NAME drag (BSX/AMD/HUM) is therefore a per-name model-fit
issue, not a per-day vol-state one — a day-level gate can't catch it and only destroys value. Interior
gates are even worse than pure bootstrap. → **Decision: ship plain GARCH-FHS ungated.** A per-NAME gate
(GARCH persistence / ARCH-LM significance) is a different, untested idea; not pursued (plain FHS already
wins and is simplest). Artifact: `output/vol_gate_experiment_2026-06-12_volgate_h21_hold63.{md,csv}`.

**Status:** GARCH-FHS ACCEPTED (plain, ungated). LIVE-default wiring into `factories.py` is the only open
step (separate deployment decision from the backtest accept).

## Session 1 single-ticker baseline (META, holdout 63, h=5, λ=0.99)
| Forecaster | mean log score | calib slope | intercept | mean PIT |
|---|---|---|---|---|
| Bootstrap (λ=0.99) | −5.062 ± 0.127 | 1.074 | −0.057 | 0.480 |
| Bootstrap (λ=0.95) | −5.120 ± 0.124 | 1.069 | −0.054 | 0.480 |
| Gaussian (λ=0.99) | −5.053 ± 0.111 | 1.039 | −0.038 | 0.481 |

Bootstrap vs Gaussian paired t (n=59): diff −0.009, t=−0.33, p=0.74 → indistinguishable. Both PIT
histograms show a mild U-shape (fat-tail miss). λ choice insignificant. (Scores differ from the
20-ticker run because that's META-only on a different window.)
