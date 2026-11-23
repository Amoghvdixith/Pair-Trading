# Bias controls

| bias | control | evidence |
|---|---|---|
| Look-ahead in selection | Screening receives only `[formation_start, cutoff]`; functions assert no date > cutoff | `test_universe_filter_rejects_future_data`, `test_formation_window_boundaries_and_future_invariance` (mutating all post-cutoff prices/volumes leaves the plan identical) |
| Look-ahead in trading | Strategy sees prices only through the backtester's point-in-time view; fills at next open | `test_future_mutation_does_not_change_past_decisions` (events and equity before a date unchanged when later prices change); fills always after the order's creation close |
| Kalman leakage | Filter only (no smoother, no full-sample EM); delta tuned by likelihood on formation data; innovation from the predicted state | `test_innovation_uses_predicted_state_only`, `test_kalman_tuning_uses_given_window_only` |
| Rolling-OLS leakage | Coefficients at t from obs t−W..t−1 | `test_rolling_ols_is_causal_and_excludes_current` |
| Liquidity/completeness look-ahead | Filters use formation data only; rejections recorded with reasons | `*_universe_rejections.csv` |
| Multiple testing | One p-value per pair (predeclared orientation), BH per formation date, BY option; raw vs adjusted vs filtered vs selected kept separate | `*_screening_audit_all_cohorts.csv`; synthetic FDP measured against truth |
| Parameter snooping | A-priori parameters for the headline; thresholds tuned only on VALIDATION cohorts; one OOS evaluation of the validation choice; the outer-test grid is labelled EXPLORATORY; every trial logged | `results/logs/trial_registry.jsonl`, `*_13_trial_log.csv` |
| Hold-out | `holdout_start` locked per dataset (`results/logs/holdout_lock.json`); hold-out rows physically removed from research data; evaluation only via `scripts/run_holdout.py --confirm`, logged | `test_holdout_requires_confirmation_and_is_excluded`; **hold-out never evaluated** |
| Overlapping cohorts | One account; each cohort limited to 1/overlap of the gross budget; one return per session | `test_cohort_capital_accounting_and_no_duplicate_returns` |
| Overlap ≠ forward validation | Adjacent-cohort persistence is reported separately from non-overlapping forward EG tests | `*_17_pair_stability.csv` |
| Survivorship (market data) | NOT controlled: current constituents only | stated in data metadata and limitations |

## Design revisions after seeing results (disclosed)

**Revision 1 (2026-11-22): `signals.zscore_lookback_min` 10 → 20.** The first full
synthetic run produced zero trades at entry z = 3.0 in the VALIDATION grid. The cause is
mathematical, not empirical: a z-score whose window includes the current point is bounded
by (L−1)/√L (2.85 at L = 10), so entry 3.0 and the 3.5 stop could never trigger. Because
synthetic half-lives are short, almost every pair sat at the floor. The floor was raised
to 20 (bound 4.25), and a validation that refuses unattainable stops was added.

*Disclosure:* the outer-test (EXPLORATORY) grid and the portfolio headline under the old
setting had already been computed when the change was made, so the synthetic outer test
cannot be called untouched with respect to this parameter. The 28 superseded trials are
recorded in `results/logs/design_revisions.jsonl` (their registry file was deleted on the
re-run, which is noted there). Market data were first run after this revision.

No threshold, cost or selection rule was changed in response to a performance number.
