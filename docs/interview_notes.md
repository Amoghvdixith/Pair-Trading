# Interview notes (answers tied to this implementation)

**1. Correlation vs cointegration?** Correlation describes co-movement of *returns*.
Cointegration means a linear combination of *price levels* is stationary. Example C in
the synthetic examples has a return correlation of 0.79 and EG p = 0.72: the levels drift
apart permanently. Example A is cointegrated (p ≈ 1e-14) with a return correlation of 0.72.

**2. Why do EG critical values differ from ADF?** The residuals come from an OLS fit that
*minimises* their variance, so they look more stationary than a raw series. The
residual-based test therefore needs MacKinnon's cointegration values (5% ≈ −3.34 for two
variables with a constant, against about −2.87 for plain ADF). The test
`test_eg_uses_cointegration_not_adf_critical_values` checks this.

**3. Testing thousands of pairs?** Restrict candidates to economic peers, use one
predeclared p-value per pair, and apply BH per formation date (BY if dependence worries
you). Then report raw, adjusted, filtered and selected counts separately. The caveat I
measured: with known truth, BH at q = 0.10 still had an FDP of 0.29, because EG p-values
are too small for "nearly cointegrated" nulls. Multiple-testing control is only as good
as the p-values fed into it. This project ran 1,845 synthetic and 2,688 market candidate
tests, not thousands per screen.

**4. Half-life of an AR(1) spread.** s_t = a + b·s_{t−1} + ε. A deviation decays as b^k,
so b^k = ½ gives k = ln 2/(−ln b) = ln 2/θ. This needs 0 < b < 1; otherwise `fit_ou`
returns a status and no half-life.

**5. Q and R in the Kalman filter?** R is the observation noise (set to the formation
OLS residual variance). Q = δ/(1−δ)·I is the random-walk variance of [α, β], with δ chosen
by innovation log-likelihood on formation data only (δ = 1e-6 in the time-varying-β
example). Because the β noise enters the innovation variance scaled by x², the innovation
z rarely exceeded 2 (0.2% of pair-days on synthetic data), so the Kalman variant traded
very little. I report that rather than tune it away. Filter only, no smoother. pykalman
agrees to about 1e-13.

**6. Pair broke mid-trade?** Every 21 sessions, EG is run on the trailing 252 sessions.
If p > 0.2, the pair is closed at the next open and suspended for the rest of its window
(an operational heuristic, not proof of a break). There is also a 3.5σ stop (and the
guarantee that it can actually be reached) and a time stop at 3 half-lives. In the
market OOS test, 4 of 32 trades ended by breakdown, 10 by time stop and 16 by take-profit.

**7. Why has simple pairs profitability declined?** I did not test this. My market
sample (2013–2023, 32 survivors) cannot separate decay from noise. The commonly cited
reasons are crowding, lower trading costs that attract competition, and faster price
discovery. I would not claim it from my own results.

**8. Truly market-neutral?** Dollar-neutral, no: the β-weighted hedge has net dollars
G(1−β)/(1+β). Market beta, measured: −0.006 (synthetic, HAC t = −1.8) and +0.003
(market, 95% CI [−0.001, 0.006]). An insignificant beta is not proof of neutrality, and
the rolling beta moves over time.

**What I would say about results.** Synthetic: the system works end to end and earns
what the DGP puts there (net excess Sharpe 1.87 OOS), but it uses only about 12% of
gross capacity. Market: no edge after costs out of sample (net excess Sharpe −0.47; the
validation-chosen thresholds went from 0.96 to −0.22), and only 8% of selected pairs were
still cointegrated a year later. Those are the honest results.
