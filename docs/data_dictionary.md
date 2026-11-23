# Data dictionary

## Inputs

| file | contents |
|---|---|
| `data/synthetic/synthetic_<mode>.csv` | long OHLCV: `date, symbol, open, high, low, close, volume` (SYNTHETIC) |
| `data/synthetic/synthetic_<mode>_universe.csv` | `symbol, sector, industry, currency, price_unit` |
| `data/synthetic/synthetic_<mode>_benchmark.csv` | synthetic market index level (`MKT`) |
| `data/synthetic/synthetic_<mode>_truth.csv` | per stock: role (`linked`/`breaking`/`unlinked`), λ, half-life, noise sd, break date |
| `data/synthetic/synthetic_<mode>_pair_truth.csv` | per same-sector pair: truly cointegrated?, until when, true log-price β |
| `data/synthetic/*.meta.json` | generator spec, seed, OHLC model, labels |
| `data/market/universe_us.csv` | author-assigned industries for 32 US stocks (input to the downloader) |
| `data/market/market_us*.csv`, `.meta.json` | yfinance cache: adjusted OHLCV, SPY benchmark, provenance (retrieval time, adjustment, missing rules, survivorship note) |

Synthetic generating model: log P_i = c_i + λ_i(M + S_k) + u_i + w_i, where M is the
market random walk, S_k the sector random walk, u_i is AR(1) noise with half-life 3–20,
and w_i is an idiosyncratic random walk (zero for linked stocks, switched on after a break
for "breaking" stocks, always on for unlinked stocks). The last sector is a pure null
sector. One stock is deliberately illiquid and one has a 60-session data gap.

## Results (prefix `syn_` = synthetic, `mkt_` = market)

| file | meaning |
|---|---|
| `results/run_metadata.json` | versions, configs, seed, data provenance, runtimes, headline numbers per data mode |
| `tables/syn_00_calibration.csv` | null rejection rates (with Wilson CIs) and estimator recovery |
| `tables/syn_01_examples_tests.csv` | tests on examples A–D |
| `tables/*_02_screening_funnel_by_cohort.csv` | funnel counts per formation date |
| `tables/syn_02b_screening_vs_truth.csv` | FDP and power of raw / BH / BY / selected vs truth |
| `tables/*_03_candidates_first_test_cohort.csv` | full candidate table (raw and adjusted p, filters, Johansen) |
| `tables/*_screening_audit_all_cohorts.csv` | every candidate test at every formation date |
| `tables/*_universe_rejections.csv` | rejected symbols with reasons |
| `tables/*_10_portfolio_metrics.csv` | gross vs net metrics (OOS, a-priori parameters) |
| `tables/*_12_cost_sensitivity.csv` | flat 0/5/10/20 bps per leg |
| `tables/*_13_trial_log.csv` | every threshold trial with its sample label |
| `tables/*_14_hedge_method_comparison.csv` | static / rolling / Kalman, gross and net |
| `tables/*_14b_oos_spread_adf_by_method.csv` | nominal ADF on trading-window spreads (see caveat) |
| `tables/*_15_neutrality_regression.csv` | market beta, HAC SE, t, CI, R² |
| `tables/*_17_pair_stability.csv` | per selected pair: re-selection, forward EG p, β drift, half-life |
| `tables/*_18_pair_trades.csv` | each completed pair trade: dates, side, exit reason, P&L, return on gross |
| `tables/*_19_cohort_exposure.csv`, `*_19_cohort_table.csv` | gross exposure by cohort; cohort dates and funnel |
| `tables/*_21_pair_attribution_reconciliation.csv` | account P&L = pairs + financing + unattributed |
| `tables/*_holdings_intended_vs_realised.csv` | entry sizing audit (intended vs rounded) |
| `tables/figure_data/*.csv` | numbers behind each figure |
| `logs/*_portfolio_net/` | full backtester audit (equity, orders, fills, status, reconciliation) |
| `logs/trial_registry.jsonl` | append-only trial registry (backtester `TrialRegistry`) |
| `logs/holdout_lock.json`, `logs/design_revisions.jsonl` | hold-out lock; disclosed design changes |

Key columns: `p_pair` (the single p-value entering FDR), `p_adj` (BH-adjusted),
`raw_reject` (p < 0.05), `fdr_reject`, `passes_filters`, `selected`, `ou_half_life`
(sessions), `gross_frac` (fraction of equity), `return_on_gross` (pair-trade P&L net of
transaction costs, excluding financing, divided by gross notional at entry).
