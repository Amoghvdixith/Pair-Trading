# Methodology

## Pipeline

```
data (synthetic | yfinance cache) ── hold-out rows removed (locked) ──┐
                                                                      ▼
cohort schedule: formation 12m → trading 6m, new cohort every 3m (fast) / 1m (research)
                                                                      ▼
per cohort, FORMATION DATA ONLY (pairs.strategy.build_cohort_plan):
  universe filters → same-industry candidates → EG (predeclared orientation, MacKinnon p)
  → BH per formation date → filters (OU valid & 2 ≤ HL ≤ 60, 0.2 ≤ β ≤ 5, ≥ 6 crossings)
  → rank by p, ≤ 1 pair per stock, ≤ 10 pairs → Kalman δ by likelihood → gross budget
                                                                      ▼
PairsStrategy (backtester.Strategy) at every close, causal:
  confirm last orders (both legs) → spread & z (static / rolling / Kalman)
  → breakdown heuristic every 21 sessions → state machine → share targets → limits
                                                                      ▼
project-02 backtester: next-open fills, costs, borrow, financing, accounting
                                                                      ▼
analysis: metrics, pair attribution, neutrality (HAC), stability, cohort audit
```

## Choices fixed before testing

| item | value | why |
|---|---|---|
| price transform | log | β is an elasticity, the spread is a log ratio; the raw alternative is documented |
| EG orientation | alphabetical (y = first symbol) | a-priori, gives one p-value per pair |
| EG deterministic term / lags | constant / AIC, statsmodels default max lag | standard |
| Johansen | det_order 0, k_ar_diff 1, 5% trace | diagnostic only |
| FDR | BH, q = 0.10, family = all candidates at one formation date | failed fits count as p = 1 |
| half-life range | 2–60 sessions | heuristic, configurable |
| z lookback | clip(1.5 × half-life, 20, 60) | 20 = floor so \|z\| can reach 4.25 (design revision 1) |
| entry / exit / stop | 2.0 / 0 (zero crossing) / 3.5 | guide defaults |
| entry mode | exceedance (\|z\| ≥ 2) + 5-session cool-down after exits | avoids immediate re-entry |
| time stop | clip(3 × half-life, 5, 120) sessions | guide default with bounds |
| breakdown | EG p > 0.20 on trailing 252 sessions, checked every 21 → close and suspend | heuristic |
| sizing | beta-weighted notional; gross ∝ 1/(P&L vol per unit gross); cap 15% per pair | comparable risk units |
| limits | gross ≤ 1.0, \|net\| ≤ 0.2, name ≤ 0.15, sector ≤ 0.5, ≤ 2 pairs per name | — |
| execution | next open; legs independent; expire if not tradable at first eligible open; unwind on leg failure | conservative legging policy |
| costs (net) | 1 bp commission + half of 4 bp spread + 1 bp slippage per leg; borrow 0.5%/yr; cash 2%, debit 4% | — |

## Samples

* **Synthetic:** data 2012-01-02 to 2021-12-31 for research; hold-out 2022-01-03 to
  2023-12-29 is locked. Validation trading windows run 2013–2016 (15 quarterly cohorts),
  OOS test trading windows 2017–2021 (19 cohorts).
* **Market:** data 2012-01-03 to 2023-12-29 for research; hold-out 2024-01-02 to
  2025-12-31 is locked. Validation 2013–2017, OOS test 2018–2023.

The validation window is used only for the threshold grid. The headline uses the
a-priori parameters on the test window. One extra evaluation of the validation-chosen
setting is reported, and the full outer-test grid is labelled EXPLORATORY.

## Modes

* `--mode fast` (default): 24 synthetic stocks (4 sectors × 6), quarterly cohorts,
  300-replication calibration.
* `--mode research`: 64 synthetic stocks (8 × 8), monthly cohorts, 1,000-replication
  calibration. **Implemented but not executed** for this report (estimated at well over
  an hour on the development laptop).
