# From an estimated spread to actual holdings

## Log-price spread (default)

s_t = log P_y,t − α − β·log P_x,t

A small change in the spread is ds ≈ r_y − β·r_x, where r is the simple return (first
order). A position with dollar notionals N_y and N_x earns about N_y·r_y + N_x·r_x. That
P&L is proportional to ds only when

**N_x = −β·N_y** (a beta-weighted *notional* hedge).

With gross exposure G = |N_y| + |N_x| and direction d (+1 = long the spread):

| quantity | formula |
|---|---|
| N_y | d·G/(1+β) |
| N_x | −d·β·G/(1+β) |
| shares | q = trunc(N/P), rounded toward zero |
| share ratio | \|q_x/q_y\| ≈ β·P_y/P_x (**not** β) |
| net dollars | N_y + N_x = d·G(1−β)/(1+β): zero only if β = 1 |

Example (unit test `test_log_beta_to_holdings`): β = 1.5, G = 100k, P_y = 50, P_x = 20
gives +800 y shares (+40k) and −3,000 x shares (−60k). Net is −20k, so 20% of gross is
dollar imbalance.

## Three different "neutralities"

| property | what makes it hold | in this project |
|---|---|---|
| spread (cointegration) hedge | N_x = −β N_y with the *right* β | yes, at entry, given the estimated β |
| dollar neutrality | N_y = −N_x | only with `notional_rule: dollar_neutral`, or if β = 1 |
| market-beta neutrality | β_mkt(y)·N_y + β_mkt(x)·N_x = 0 | not targeted; *measured* ex post by regression |

None of the three implies the others, and the project does not claim all three at once.

## Raw-price spread (alternative)

s = P_y − α − β·P_x. Here β **is** a share ratio: hold q_y shares of y and −β·q_y
shares of x (`raw_price_holdings`). This interpretation is documented but not used in
the experiments.

## Rules

* **Intercept.** α is a level offset. It is never traded.
* **Beta policy (predeclared).** Trade only finite β in [0.2, 5]. Pairs outside this
  range are rejected at screening, and an entry is skipped if the live β (rolling or
  Kalman) is outside it.
* **Gross normalisation.** Each pair's gross is a fraction of current equity, allocated
  in proportion to 1/(daily P&L sd per unit gross) using formation returns. It is capped
  at 15% per pair and fits inside the cohort's share of the 100% portfolio gross cap.
  Pairs whose allocation would fall below 2% are not traded.
* **Admission control.** A new pair is scaled down (minimum 25%) or rejected so that
  portfolio gross ≤ 1.0, |net| ≤ 0.20, |w| ≤ 0.15 per stock, sector gross ≤ 0.50, and at
  most 2 live pairs per stock. Existing positions are never resized by admission.
* **Rounding.** Shares are rounded toward zero. Intended and realised notionals, dollar
  imbalance and the rounding gap are logged for every entry
  (`*_holdings_intended_vs_realised.csv`).
* **Changing β while open.** Static OLS never re-hedges. Rolling OLS and Kalman re-size
  the x leg when |β_t/β_entry − 1| > 10% and keep the y leg. Every re-hedge is a real
  order through the backtester and pays commission, spread and slippage. (Synthetic
  OOS run: 187 confirmed re-hedges for rolling OLS; their costs are inside its total.)
* **P&L.** All P&L comes from actual fills, prices, cash flows and costs in the
  backtester. It is never computed as the change in a re-estimated residual. Pair-level
  P&L is reconstructed from confirmed pair holdings and actual fill prices, and
  reconciles to the account: account P&L = Σ pair P&L + financing + unattributed, with
  |unattributed| below 1e-8 in every recorded run.
