# Statistical validation and corrections to the build guide

This document lists (1) where the supplied build guide (`03_Pairs_Trading_Guide.pdf`) was
corrected or qualified, (2) how each formula was verified, and (3) the inference limits
that remain. Numbers quoted come from `results/tables/syn_00_calibration.csv` and the
other result files produced by `scripts/run_all.py`.

**How formulas were verified.** No web browsing was used for this project. Formulas are
taken from the cited literature and verified *numerically*: against statsmodels
(`coint`, `adfuller`, `coint_johansen`, `mackinnoncrit`), against pykalman (identical
model, max abs state difference about 1e-13), by hand-calculated unit tests, and by
simulation under known truth (null rejection rates with Wilson intervals). Citations are
from the literature as commonly published. Check page-level details yourself before
quoting them.

## 1. Corrections and qualifications

| # | Guide says | Problem | What this project does |
|---|---|---|---|
| 1 | Engle–Granger: "run both directions and keep the pair only if both pass, **or choose the direction with the more negative statistic**" | Taking the better of two p-values without adjustment inflates the false-positive rate. It also does not give one valid p-value per pair for an FDR procedure. | The regression orientation is fixed **before** testing (`alphabetical` by default, or `max_variance`), giving one p-value per pair. Optional `both_bonferroni`: p_pair = min(1, 2·min(p_yx, p_xy)), which is valid by Bonferroni. Both directional p-values are always kept in the audit table. |
| 2 | "two independent random walks are rejected in about 95% of seeds" | Wording reversed. Under H0 (no cointegration), the test should **reject about 5%** of the time. Not rejecting is not proof of no cointegration. | Calibration: 5.3% [3.3%, 8.5%] at n=252 and 4.7% [2.8%, 7.7%] at n=1000 (300 seeds, Wilson 95%). |
| 3 | Pass criteria "β̂ within 0.05 of truth; EG rejects at 1%" and "half-life within 15% at 2,000 obs" **per test sample** | A single noisy sample cannot be required to hit these targets. Doing so makes tests seed-dependent. | Unit tests use loose single-sample bounds. Bias and dispersion are reported across 300 replications: β̂ bias 0.001, sd 0.031, 89% within 0.05; half-life mean 9.90 (true 10), 78% within 15%. |
| 4 | `fit_ou` code: `theta = -np.log(b)` with no guard | For b ≤ 0 or b ≥ 1 this returns NaN or a negative, plausible-looking half-life. | `fit_ou` returns status `negative_b`, `unit_root_or_explosive`, `constant` or `insufficient_data` and **no** half-life. It also gives a half-life CI from the OLS interval for b. |
| 5 | OU: σ_eq = sd(ε)/√(1−b²) | This is correct, but it is the *stationary* sd of the spread level. It is not the diffusion σ. | Both are reported: `stationary_sd` and `sigma_diffusion = sd(ε)·√(2θ/(1−b²))`. |
| 6 | Trading table: "Short A, long β units of B **(dollar-neutral)**" | With log prices, β is an elasticity, not a share count. "β units" is a beta-weighted *notional* hedge, which is dollar-neutral only if β = 1. | Holdings use N_x = −β·N_y with \|N_y\|+\|N_x\| = G. Net dollar exposure = G(1−β)/(1+β) is reported (up to 66% of gross on synthetic data, 35% on market data). A `dollar_neutral` rule is available as an alternative. See `hedge_to_holdings.md`. |
| 7 | z lookback L ≈ 1–2 half-lives with a stop at \|z\| ≥ 3.5 | If the current value is inside its own window, Samuelson's inequality gives \|z\| ≤ (L−1)/√L. For L < 15 a 3.5 stop **can never trigger**, and for L = 10 neither can entry at 3.0. The first version of this project hit exactly this bug (see `bias_controls.md`, design revision 1). | The lookback floor is 20 (\|z\|max = 4.25). `StrategyParams` refuses configs whose stop is unattainable. Unit test `test_zscore_bound_with_current_included`. |
| 8 | Kalman: "e_t/√F_t is itself a ready-made z-score" | It is N(0,1) only if the model is right. The spread is autocorrelated and the state adapts, so the filter absorbs divergence. With Q = δ/(1−δ)·I, the β-noise enters F scaled by x² (log price ≈ 3–5), which inflates F. | Innovation and posterior residuals are reported separately. Observed: Kalman innovation \|z\| ≥ 2 on only 0.2% (synthetic) and 2.3% (market) of pair-days, so the Kalman variant trades rarely. Reported as a finding, not tuned away. |
| 9 | Kalman covariance update P = (I−KH)P_pred | Algebraically correct, but it can lose symmetry or PSD in floating point. | Joseph form plus symmetrisation. The test checks symmetry and min eigenvalue ≥ −1e-14 over 3,000 steps. pykalman matches to 1e-8 (test) and about 1e-13 (example). |
| 10 | Johansen: "the eigenvector for the largest eigenvalue gives the cointegrating weights directly" | The weights are identified only up to scale, and are an estimate, not a validated basket. The trace statistics are compared with tabulated critical values, not p-values. | Normalised so the y-weight is 1 (`beta_johansen = −v2/v1`). Used as a diagnostic only, never in the FDR family. Rank 2 (full rank) is labelled "both series look I(0)". Null calibration: rank ≥ 1 in 8.7% [6.0%, 12.4%] of independent-RW samples at nominal 5%, so it is somewhat oversized with `det_order=0`. |
| 11 | BH-FDR "controls the false-discovery rate" | BH controls E[FDP] ≤ q·m0/m within **one family**, if p-values are valid and the tests are independent or PRDS. Pairs that share a stock are dependent. Repeating the screen every quarter is not covered. | The family is defined per formation date. BY (valid under any dependence) is available. Per-window control is never described as global. See §3 for the **observed failure of FDR control** on synthetic data. |
| 12 | Break-down stop: "rolling coint p-value > 0.2 → close and drop" | Failing to reject is not proof of a break. Re-testing every month is repeated sequential testing with no error-rate guarantee. | Kept as an **operational heuristic** (every 21 sessions, 252-session trailing window, EG p > 0.2). It is labelled that way everywhere. |
| 13 | Neutrality: "Target \|β\| < 0.1" | An insignificant beta is not proof of neutrality. Beta also varies over time. | The report gives β, HAC (Newey–West) SE, t, 95% CI and R², plus a rolling 126-day beta. Beta is not a pass/fail test. |
| 14 | "Stability typically falls after crises"; "the edge has shrunk since the 2000s" | These are empirical claims. Assuming them is a form of look-ahead bias. | This project only reports what its own data show. The market sample (2013–2023) is too short and survivor-biased to test either claim. |
| 15 | Equal risk: "scale the gross position by 1/σ_spread" | Residual σ from different hedge models, or log versus raw prices, is not comparable P&L risk. | Gross is budgeted by 1/(daily P&L sd per unit gross of the hedged position), using formation returns, with floor and cap. Correlation between pairs is ignored, and this is documented. |
| 16 | pyproject depends on `backtester @ git+https://…`; `.github/workflows/ci.yml` | The user excluded Git operations and CI configuration. | The backtester wheel is built from a local copy (`scripts/install_local_backtester.py`). No CI files are created. |
| 17 | Step 6: 5 bp commission + half spread + 0.5% borrow | — | The default "net" model is 1 bp commission + half of a 4 bp spread + 1 bp slippage per leg, with 0.5%/yr borrow. The flat 5/10/20 bp scenarios bracket the guide's assumption. |

## 2. Verified formulas

* **Engle–Granger** (Engle & Granger, 1987). Step 1: OLS y = α + βx. Step 2: ADF on the
  residuals with AIC lag selection, using response-surface p-values for residual-based
  tests with N = 2 (MacKinnon, 1994 and 2010; Phillips & Ouliaris, 1990 for the
  asymptotics). The 5% critical value is about −3.34 (constant, N = 2). Tested against
  `mackinnoncrit` and compared with the plain ADF value (about −2.87).
* **Johansen** (Johansen, 1991): `statsmodels.tsa.vector_ar.vecm.coint_johansen`.
  Settings: `det_order=0`, `k_ar_diff=1`, sequential trace test at the tabulated 5% level.
* **BH** (Benjamini & Hochberg, 1995): step-up with adjusted p-values
  p̃(i) = min_{j≥i} m·p(j)/j. **BY** (Benjamini & Yekutieli, 2001): multiply by
  c(m) = Σ 1/i. Checked by hand examples and by simulation (realised FDR 0.076 against a
  bound of 0.08 for 40 nulls and 10 alternatives, q = 0.10).
* **OU ↔ AR(1)** (exact discretisation): b = e^{−θΔt}, μ = a/(1−b), half-life = ln2/θ,
  stationary sd = sd(ε)/√(1−b²), σ = sd(ε)√(2θ/(1−b²)). Tested algebraically and for dt
  scaling.
* **Variance ratio** (Lo & MacKinlay, 1988): overlapping, bias-corrected, with a
  heteroskedasticity-robust z* = √T·M/√θ̂. A smoke test caught a missing √T scale during
  development; it was fixed before any result was written. Null calibration (t(5)
  increments): robust z 5.7% [3.6%, 8.9%], homoskedastic z 6.7%.
* **Hurst**: slope/2 of log Var(s_{t+τ} − s_t) on log τ over τ = 2..20. This is
  exploratory only; it is biased toward 0 for stationary series as τ grows.
* **Kalman filter** (Kalman, 1960; Joseph-form update, e.g. Bucy & Joseph, 1968;
  Durbin & Koopman, 2012): predict m⁻ = m, P⁻ = P + Q; innovation e = y − Hm⁻;
  F = HP⁻H′ + R; K = P⁻H′/F; update. The innovation is taken before the update. Tested
  by hand calculation and against pykalman with the identical prior (pykalman starts
  without a transition, so it receives P0 + Q).
* **Samuelson bound** (Samuelson, 1968): for a sample of size L, any point satisfies
  \|x − x̄\|/s ≤ (L−1)/√L. Tested.
* **Neutrality regression**: OLS with Newey–West HAC standard errors (Newey & West, 1987),
  5 lags daily and 4 lags weekly.

## 3. Inference limitations observed in this project's own runs

1. **FDR was *not* controlled on the synthetic universe.** The data-generating truth is
   known, so the false-discovery proportion can be measured: mean FDP across 34 formation
   dates was 0.29 for BH at q = 0.10 and 0.10 for BY (with far lower power: 3.8% versus
   22%). Raw EG p < 0.05 had an FDP of 0.50. The cause is that the EG p-values are
   **anti-conservative for "nearly cointegrated" nulls**: unlinked stocks share a sector
   factor and carry only a small idiosyncratic random walk next to large stationary noise.
   Over 12 months the unit-root component is hard to see: in the calibration, EG at
   nominal 5% rejected 14.3% (RW sd 0.009, the universe's value) and 38% (RW sd 0.003)
   of 300 samples whose residual truly contains a random walk. BH cannot fix invalid p-values. This is the most
   important statistical lesson of the project.
2. **Adaptive spreads.** ADF p-values on rolling-OLS or Kalman spreads (median nominal
   p = 0.07 and 0.0008 on synthetic data) are not valid tests. The coefficients were
   re-estimated with information from the same sample, so the residuals look stationary
   by construction.
3. **Repeated screening.** Every quarter re-tests largely the same pairs on overlapping
   windows. Per-window FDR gives no guarantee across the sequence of screens. Persistence
   between adjacent windows (overlapping data) is reported separately from
   non-overlapping forward tests.
4. **Rolling breakdown tests** are sequential and multiplicity-uncorrected. They are
   heuristics.
5. **Short-window power.** At n = 252, EG power is 96% for half-life 5, 55% for
   half-life 10 and 10% for half-life 30. A 12-month formation window cannot detect slow
   mean reversion.
6. **Threshold tuning** on a single validation split with 12 to 15 configurations has
   high selection noise. On market data the validation winner (Sharpe 0.96) delivered
   −0.22 out of sample.

## References

Benjamini, Y. & Hochberg, Y. (1995) JRSS B 57(1). Benjamini, Y. & Yekutieli, D. (2001)
Annals of Statistics 29(4). Bucy, R. & Joseph, P. (1968) *Filtering for Stochastic
Processes with Applications to Guidance*. Durbin, J. & Koopman, S.J. (2012) *Time Series
Analysis by State Space Methods*, 2nd ed. Engle, R. & Granger, C. (1987) Econometrica
55(2). Gatev, E., Goetzmann, W. & Rouwenhorst, K. (2006) RFS 19(3). Johansen, S. (1991)
Econometrica 59(6). Kalman, R. (1960) J. Basic Eng. 82(1). Lo, A. & MacKinlay, A.C.
(1988) RFS 1(1). MacKinnon, J. (1994) JBES 12(2); (2010) Queen's Economics WP 1227.
Newey, W. & West, K. (1987) Econometrica 55(3). Phillips, P. & Ouliaris, S. (1990)
Econometrica 58(1). Samuelson, P. (1968) American Statistician 22(5). Uhlenbeck, G. &
Ornstein, L. (1930) Physical Review 36. Chan, E. (2013) *Algorithmic Trading*.
Vidyamurthy, G. (2004) *Pairs Trading*.
