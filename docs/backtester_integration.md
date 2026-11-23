# Integration with the project-02 backtester

## Status

| item | status |
|---|---|
| Located | `<Desktop>/backtester` (sibling folder), package `backtester` 0.1.0 |
| Inspected | `engine.py`, `strategy.py`, `sizing.py`, `execution.py`, `costs.py`, `portfolio.py`, `data.py`, `results.py`, `performance.py`, `walkforward.py` |
| Its own tests | 93 passed (2 slow deselected) against the installed wheel, run from a scratch copy of its `tests/` (no files written in its folder) |
| Installed how | `scripts/install_local_backtester.py`: copies the source to a temp dir, builds a wheel, installs it. No Git, no editable install, nothing written into the original folder |
| Modified | **No.** Nothing in project 02 was edited |
| Integration | **Working** for the adapter below, exercised by `tests/test_backtest_integration.py` and every experiment |

## Public interfaces used

`Backtest`, `BacktestConfig`, `MarketData.from_long`, `Strategy` (subclassed by
`PairsStrategy`), `Sizer` (subclassed by `PassThroughSizer`), `PortfolioConstraints`,
`CostModel` / `Commission` / `SpreadCost` / `Slippage`, `FinancingModel`,
`ExecutionConfig`, `BacktestResult` (equity, fills, order status, reconciliation),
`compute_metrics`, `backtester.performance.rf_from_annual`, and
`backtester.walkforward.{check_holdout_lock, TrialRegistry, HoldoutViolation}`.

## Conventions inherited (not re-implemented)

* Signal at the close of t; orders are eligible from the next session's open and filled
  at the first tradable open (`expires_after=0` → cancelled if that open is not
  tradable).
* Sells/shorts are processed before buys; whole shares; buys are resized pro rata if
  cash is short.
* Commission is a cash fee; spread, slippage and impact are embedded in the fill price
  (no double counting).
* Financing accrues daily on calendar days: credit on cash excluding short proceeds,
  debit on negative cash, borrow on |short value|.
* Accounting identities are reconciled at every close; FIFO lots; terminal liquidation
  at the final open.

## How the adapter works

1. `PairsStrategy.on_bar` (called at each close) confirms the previous orders, updates
   each pair's state machine, and emits the **whole stock book** as `SignalEvent`s whose
   `direction` is a target weight.
2. `PassThroughSizer` passes those weights through unchanged. `PortfolioConstraints` is a
   *loose backstop* (gross 1.25×cap, name 2×cap); the strategy enforces the real limits
   itself. Each run records `backstop_bound` (whether the backstop ever changed a
   weight). It was `False` in every recorded run.
3. The engine turns weight w into `trunc(w·equity/close)` shares. The adapter passes
   w = sign(q)(|q|+0.5)·close/equity, so the engine reproduces exactly the intended
   integer q despite floating-point truncation (tested on 2,000 random cases).
4. Stock-level orders are **netted across pairs and cohorts** by construction (one target
   per symbol). Pair attribution is kept in the strategy's ledger, not by inventing
   separate stock holdings.
5. Fill confirmation: at the next close, actual positions are compared with expected
   positions symbol by symbol. If any leg of a pending entry mismatches, that is a leg
   failure: the pair's target goes back to zero and the filled leg is unwound at the next
   open (deterministic policy). Tested by deleting one leg's bar on the fill day.

## Change requests (CR) for project 02, not applied

* **CR-1, portfolio snapshot for strategies.** `Strategy` receives only a
  `PointInTimeView`. Pair trading needs current equity (for sizing) and post-fill
  positions (for two-leg confirmation), so the adapter reads `Backtest.portfolio` through
  a callable. Proposal: pass a read-only `PortfolioSnapshot(equity, cash, positions,
  marks)` to `on_bar`, or expose it on the view.
* **CR-2, share-target orders.** Add an optional `target_shares` path (or
  `SignalEvent.target_quantity`) so strategies that size in shares do not need the
  +0.5-share weight encoding.
* **CR-3, linked / contingent orders.** Native "all-or-none" or one-cancels-other
  groups for multi-leg trades would allow modelling atomic pair fills. Today legs are
  independent, so legging risk exists and is handled by the unwind policy.
* **CR-4, per-symbol borrow fees and locate availability.** `FinancingModel.borrow_fee`
  is one rate for every short. Hard-to-borrow names need per-symbol rates and a locate
  check.
* **CR-5, `pyproject.toml` license field.** Setuptools warns that the TOML-table license
  form is deprecated (it will stop building after 2027-02-18). Switch to an SPDX string.
