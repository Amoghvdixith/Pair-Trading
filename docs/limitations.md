# Limitations

**Statistical**
* EG p-values are asymptotic. They are anti-conservative when the null is "nearly
  cointegrated" (measured: 14–38% rejections at nominal 5%), so BH did not control FDR on
  the synthetic universe. BY is more robust but has very low power.
* A 12-month window has little power for half-lives above about 10 sessions.
* Johansen with `det_order=0` is mildly oversized (8.7% at nominal 5%). It is a
  diagnostic only.
* Rolling-window breakdown tests, repeated screens and the threshold grid are not covered
  by any global error-rate guarantee.
* ADF on adaptive (rolling or Kalman) spreads is not valid inference.
* OU half-lives come from OLS AR(1) fits, which are biased toward faster mean reversion
  in short samples. Intervals are large-sample approximations.

**Trading and accounting**
* Daily bars only. Stops and exits execute at the next open, so gaps can carry the fill
  far beyond a threshold. No intraday or limit orders.
* Legs are filled independently (not atomic). The unwind policy limits but does not remove
  legging risk.
* Borrow is a single 0.5%/yr general-collateral rate. Hard-to-borrow names, recalls and
  locate failures are not modelled (change request CR-4).
* Cash earns 2% and short proceeds earn nothing. Real rebates and margin terms vary.
  Because net CAGR includes interest, the reported Sharpe uses excess returns.
* Market data are split- and dividend-adjusted closes and opens (a total-return
  approximation). Share counts are in adjusted units; dividends on shorts and
  corporate actions are not modelled as cash events.
* Capacity, market impact and short-sale constraints are not modelled beyond the cost
  model.
* Capital is under-used. Average gross exposure was 12% (synthetic) and 3% (market),
  because pairs are capped at 15% and most of the time no pair is in a trade.

**Data and universe**
* The market universe is 32 *current* large caps with author-assigned industries.
  Survivorship bias and look-ahead in membership are present, and there is no
  point-in-time classification history. The optional `membership` hook exists but no
  historical membership data were used.
* yfinance is an unofficial API; revisions to the data are possible. The cached data are
  for local research only.
* Synthetic results show that the code works as designed. Their Sharpe ratios come from
  a data-generating process built to contain mean reversion and are not evidence of
  market alpha.

**Not implemented / unfinished**
* Research mode was not executed.
* The hold-out was **not** evaluated (by design).
* Stretch goals from the guide (distance method, PCA residuals, copulas, regime filters,
  Bertram optimal thresholds, intraday) are not implemented.
* No deflated Sharpe or block-bootstrap confidence intervals for this project's Sharpe
  ratios (the backtester offers them; not wired in).
* mypy reports remaining errors (mostly pandas-stubs strictness); see the README.
