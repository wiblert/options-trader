# Research note — improving forecast accuracy (2026-06-12)

Deep web-research pass (fan-out search → fetch 21 sources → 3-vote adversarial verification →
synthesis; 97 claims extracted, 25 verified, 0 refuted). Triggered by the Session-11 finding that
**additive drift-dampening gave no aggregate log-score lift**. Goal: where to spend the next
forecaster experiment to raise mean log score, split by what it fixes — CENTER vs VOLATILITY LEVEL vs
TAILS/SHAPE.

## TL;DR
- **The CENTER/drift question is effectively settled — and our null result is exactly what the
  literature predicts.** The trailing mean return is near-unestimable; shrinking it toward an economic
  anchor is the right default; but the gain is *small at short horizons*. So our dampening experiment
  failing to move the log score is consistent with theory, not a surprise. Stop spending on the center.
- **The binding constraint is TAILS/SHAPE (our PIT U-shape) and VOLATILITY LEVEL.** That's where the EV
  is. The single highest-value experiment is **Filtered Historical Simulation (FHS)** — it fixes fat
  tails *and* vol clustering in one move and is a direct upgrade to our iid bootstrap.
- **Confidence asymmetry to keep honest:** the drift/center conclusions are strongly verified (Merton,
  Goyal-Welch, Campbell-Thompson, PTV, Hjalmarsson). The specific tail/vol *techniques* (FHS, Student-t,
  GARCH, HAR-RV) were **not** independently verified in this batch — they rest on standard priors, so
  each must clear our OOS log-score paired test. The one hard verified result on the vol/tail side is a
  *caution* (Carr-Wu single-name VRP), not an endorsement.

## 1. The drift / center question — SETTLED (low EV to keep pushing)
- **Merton (1980, JFE):** the sample-mean estimator's variance is σ²/h — depends ONLY on the total
  calendar span h, never on sampling frequency. "Nothing is gained in accuracy of the expected-return
  estimate by choosing finer observation intervals." Variances are far more estimable than means. → the
  statistical reason trailing single-name drift is mostly noise. [verified 3-0]
- **Goyal & Welch (2008):** "none of the popular variables has worked" OOS against the historical mean.
  **Pettenuzzo-Timmermann-Valkanov (2014):** 12/16 unconstrained predictors give negative OOS R²
  (avg −0.53%/mo). [verified 3-0]
- **Hjalmarsson (2006, Fed IFDP 855):** when a slope is small and noisy "you are often better off
  setting it equal to zero"; OOS predictability tests have very low power — **a null OOS lift does not
  prove a signal useless.** → directly explains our dampening result: per-ticker drift is a genuine but
  *undetectable* signal+noise mix, so a uniform κ nets to zero. [verified 3-0]
- **Campbell-Thompson (2008) + PTV:** the right fix for the center is not a better drift estimate but an
  **economic anchor** — sign/steady-state restrictions "never worsen and almost always improve" OOS,
  precisely because they "remove the need to estimate the average from a short volatile sample." Gains
  grow with horizon but are small in absolute terms (sub-1% monthly OOS R²). [verified 3-0]
- **Spot vs forward anchor:** NOT directly resolved by any verified source. At 5–21 days the carry term
  (r−q)·T is tiny vs terminal dispersion, so spot and the no-arb forward are nearly indistinguishable
  for log score. The decision that matters is *shrinking toward an anchor*, not *which* anchor.
  [medium confidence — inference from carry magnitude]

## 2. Where the EV is — TAILS/SHAPE and VOLATILITY LEVEL (prioritized)
Mapped to our backlog and gated by the standing log-score criterion. Evidence confidence noted.

- **[P0 — TAILS + VOL] Filtered Historical Simulation (FHS).** GARCH(1,1)-filter the returns → bootstrap
  the *standardized* residuals → re-inflate by the volatility forecast over the horizon. Captures fat
  tails AND vol clustering simultaneously; it is the non-parametric upgrade of our current iid bootstrap
  and the most direct attack on the PIT U-shape. *Top open question from the research.* (Technique
  unverified in-batch → backtest it.)
- **[P0 — TAILS] Fat-tail innovations** (Student-t / NIG / variance-gamma, optionally with EGARCH/GJR for
  asymmetry). This is already our #1 backlog item; FHS is essentially its non-parametric sibling. Open
  question: which family + what tail calibration maximizes log score/CRPS on single names.
- **[P1 — VOL LEVEL] Time-varying vol scaling** — GARCH family, HAR-RV (Corsi) on realized vol, or EWMA.
  Even short of full GARCH, scaling the bootstrap draws by a time-varying vol forecast is cheap; note our
  λ=0.99 decay is already a crude EWMA, so this is an incremental, testable step up.
- **[P1 — VOL LEVEL, forward-looking] Option-implied variance, handled carefully for single names.**
  **Carr-Wu (2009, RFS):** the single-name variance risk premium is small/insignificant at the LEVEL
  (only 3/35 stocks significant) — the market prices variance risk mainly at the *index* level. BUT
  implied variance is a largely *unbiased* predictor of realized variance up to a roughly constant **log**
  premium (log-VRP well-behaved for ~21-24/35 names). → Feed option-implied variance into the vol level,
  subtract a *calibrated, roughly-constant, per-name log* VRP offset, and gate on log score. Do NOT apply
  an index-style level adjustment per name. [verified 3-0 — this is the one hard vol-side result]
- **[P2 — CENTER, low EV] Forward-anchor test.** Confirm spot ≈ forward at 5–21d, treat the center as
  settled (shrink toward anchor, expect a small effect). Don't reinvest here beyond a confirmation run.
- **[P2 — robustness] Cross-sectional James-Stein pooling** of vol/drift params across the 20-name set
  for short-history names, and a **forecast-combination ensemble** (bootstrap + GARCH-FHS + implied-vol)
  once the components exist.

## Recommended next experiment
**Build a GARCH-filtered FHS forecaster and backtest it vs the bootstrap incumbent on the standardized
set.** It is the single change with the strongest claim on our documented residual miss (fat tails +
vol clustering), and the project already has a working GARCH(1,1) MLE implementation to borrow from (in
the sibling quant-research repo). Gate strictly on mean log score, paired, per the decision criterion.

## Open questions the research flagged (each a backtestable experiment)
1. Does FHS beat plain weighted-iid bootstrap on OOS mean log score for single names at 5–21d?
2. Among fat-tail innovations (t vs NIG vs variance-gamma) × {EGARCH, GJR}, which maximizes log
   score/CRPS, at what tail calibration?
3. Does option-implied variance minus a calibrated constant *log* VRP beat HAR-RV/realized-vol for the
   physical density's vol level on single names?
4. Does James-Stein pooling across the 20 names help short-history names, and does an ensemble beat the
   best single forecaster?

## Sources (verified subset)
Merton 1980 JFE (rotman.utoronto.ca PDF); Goyal & Welch 2008 (NBER w11468); Pettenuzzo-Timmermann-Valkanov
2014 (rady.ucsd.edu PDF); Campbell-Thompson 2008 / Campbell 2008; Hjalmarsson 2006 (Fed IFDP 855);
Carr & Wu 2009 RFS (engineering.nyu.edu PDF); Fed IFDP 1035 (model-free VRP). Full list +
verification verdicts in the workflow journal (run wf_fdff7ff2-ed7).
