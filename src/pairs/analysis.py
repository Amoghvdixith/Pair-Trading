"""Evaluation: metrics, pair attribution, neutrality, stability and cohort audits.

Gross vs net (precise definitions used everywhere in this project)
------------------------------------------------------------------
* GROSS run: identical signals and sizing rules, but no commission, spread, slippage,
  borrow fee or interest (pure price P&L).  Sharpe uses rf = 0.
* NET run: detailed transaction costs on both legs (entry, exit, rehedge, legging
  unwinds) + 0.5%/yr borrow on |short value| + interest on cash (credit 2%, debit 4%,
  short proceeds earn nothing).  Sharpe uses excess returns over rf = credit rate,
  accrued on calendar days exactly like the cash account.
Gross and net are separate backtests; share counts can differ slightly because equity
differs, but trade timing does not (signals depend on prices only).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd
import statsmodels.api as sm
from backtester import BacktestResult, MarketData, compute_metrics
from backtester.performance import rf_from_annual

from .cohorts import Cohort
from .data import log_prices
from .ou import fit_ou
from .screening import ScreenConfig, engle_granger
from .strategy import CohortPlan, PairsStrategy


# --------------------------------------------------------------------------- metrics
def performance(res: BacktestResult, rf_annual: float, start: pd.Timestamp | None = None,
                end: pd.Timestamp | None = None, benchmark: pd.Series | None = None,
                round_trips: pd.DataFrame | None = None) -> dict[str, Any]:
    """Backtester ``compute_metrics`` on the window [start, end] (default: trading window)."""
    r = res.returns(start, end)
    rf = rf_from_annual(r.index, rf_annual) if rf_annual else None
    bench = benchmark.pct_change().reindex(r.index) if benchmark is not None else None
    return compute_metrics(r, rf=rf, benchmark=bench, equity_frame=res.equity,
                           round_trips=round_trips)


# --------------------------------------------------------------------------- attribution
def attribute_pairs(res: BacktestResult, strategy: PairsStrategy, data: MarketData
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair-level cumulative P&L from confirmed pair holdings and ACTUAL fills.

    Each confirmed pair share change (ledger, dated on its fill session) is valued at that
    session's fill price for the symbol; if the engine netted the change away (opposite
    changes in another pair) the open price is used and no commission is charged.
    Commission of a fill is shared pro rata to |pair change|.  Positions are marked at
    closes (last observed close if missing, like the engine).  Financing is NOT
    allocated to pairs.  Returns (cumulative P&L per pair key, reconciliation frame):

        engine P&L = sum(pair P&L) + net financing + unattributed

    ``unattributed`` captures legging (filled leg of a failed pair) and nothing else in a
    clean run; it is reported, not hidden.
    """
    idx = res.equity.index
    led = pd.DataFrame(strategy.ledger)
    eq = res.equity
    fin = (eq["interest"] - eq["borrow_fees"]).reindex(idx)
    total = eq["equity"] - float(strategy._pf().initial_cash)
    if led.empty:
        rec = pd.DataFrame({"engine_pnl": total, "pairs_pnl": 0.0, "financing": fin})
        rec["unattributed"] = rec["engine_pnl"] - rec["pairs_pnl"] - rec["financing"]
        return pd.DataFrame(index=idx), rec
    fills = res.fills.copy()
    if not fills.empty:
        fills["timestamp"] = pd.to_datetime(fills["timestamp"])
        fmap = {(r.timestamp, r.symbol): (r.fill_price, r.commission)
                for r in fills.itertuples()}
    else:
        fmap = {}
    opens = data.frame("open")
    closes = data.frame("close").reindex(idx).ffill()
    led["date"] = pd.to_datetime(led["date"])
    tot_abs = led.groupby(["date", "symbol"])["delta"].apply(lambda s: s.abs().sum())
    cash_rows = []
    for r in led.itertuples():
        px, comm = fmap.get((r.date, r.symbol), (float(opens.at[r.date, r.symbol]), 0.0))
        share = abs(r.delta) / tot_abs[(r.date, r.symbol)]
        cash_rows.append({"date": r.date, "key": r.key,
                          "cash": -r.delta * px - comm * share})
    cash = (pd.DataFrame(cash_rows).groupby(["date", "key"])["cash"].sum()
            .unstack("key").reindex(idx).fillna(0.0).cumsum())
    pos = (led.groupby(["date", "key", "symbol"])["delta"].sum().unstack(["key", "symbol"])
           .reindex(idx).fillna(0.0).cumsum())
    value = pd.DataFrame(0.0, index=idx, columns=cash.columns)
    for key, sym in pos.columns:
        value[key] += pos[(key, sym)] * closes[sym]
    pnl = value + cash
    rec = pd.DataFrame({"engine_pnl": total, "pairs_pnl": pnl.sum(axis=1), "financing": fin})
    rec["unattributed"] = rec["engine_pnl"] - rec["pairs_pnl"] - rec["financing"]
    return pnl, rec


def trade_table(strategy: PairsStrategy, pair_pnl: pd.DataFrame) -> pd.DataFrame:
    """Completed pair trades with P&L (net of transaction costs, excl. financing)."""
    tr = pd.DataFrame(strategy.trades)
    if tr.empty:
        return tr
    tr = tr[tr.get("exit_fill").notna()] if "exit_fill" in tr else tr.iloc[0:0]
    rows = []
    aud = pd.DataFrame(strategy.holdings_audit)
    for r in tr.to_dict("records"):
        s = pair_pnl[r["key"]] if r["key"] in pair_pnl else None
        pnl = math.nan
        if s is not None:
            before = s[s.index < r["entry_fill"]]
            p0 = float(before.iloc[-1]) if len(before) else 0.0
            pnl = float(s.loc[r["exit_fill"]]) - p0
        g = aud[(aud["key"] == r["key"]) & (aud["date"] < r["entry_fill"])]["gross"]
        gross = float(g.iloc[-1]) if len(g) else math.nan
        rows.append({**r, "pnl": pnl, "gross_at_entry": gross,
                     "return_on_gross": pnl / gross if gross else math.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- neutrality
def neutrality_regression(r: pd.Series, mkt: pd.Series, hac_lags: int = 5,
                          rf: pd.Series | None = None) -> dict[str, Any]:
    """OLS of (excess) strategy returns on (excess) market returns with Newey-West SEs.

    An insignificant beta is NOT proof of neutrality (low power, time variation); report
    the confidence interval and look at the rolling estimate too.
    """
    df = pd.concat({"r": r, "m": mkt}, axis=1).dropna()
    if rf is not None:
        rr = rf.reindex(df.index).fillna(0.0)
        df["r"], df["m"] = df["r"] - rr, df["m"] - rr
    if len(df) < 30:
        return {"n": len(df), "status": "insufficient data"}
    X = sm.add_constant(df["m"])
    fit = sm.OLS(df["r"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    ci = fit.conf_int().loc["m"]
    return {"status": "ok", "n": int(len(df)), "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()), "frequency": "daily",
            "beta": float(fit.params["m"]), "beta_se_hac": float(fit.bse["m"]),
            "beta_t": float(fit.tvalues["m"]), "beta_ci_low": float(ci.iloc[0]),
            "beta_ci_high": float(ci.iloc[1]), "alpha_daily": float(fit.params["const"]),
            "alpha_t": float(fit.tvalues["const"]), "r2": float(fit.rsquared),
            "cov": f"HAC Newey-West, {hac_lags} lags"}


def rolling_beta(r: pd.Series, mkt: pd.Series, window: int = 126) -> pd.Series:
    df = pd.concat({"r": r, "m": mkt}, axis=1).dropna()
    cov = df["r"].rolling(window).cov(df["m"])
    var = df["m"].rolling(window).var()
    return (cov / var).rename("rolling_beta")


# --------------------------------------------------------------------------- cohorts
def cohort_exposure(strategy: PairsStrategy, data: MarketData, index: pd.DatetimeIndex,
                    equity: pd.Series) -> pd.DataFrame:
    """Daily gross exposure (fraction of equity) per cohort from CONFIRMED pair holdings."""
    led = pd.DataFrame(strategy.ledger)
    out = pd.DataFrame(index=index)
    if led.empty:
        out["total"] = 0.0
        return out
    led["date"] = pd.to_datetime(led["date"])
    led["cohort"] = led["key"].str.split(":").str[0].astype(int)
    closes = data.frame("close").reindex(index).ffill()
    pos = (led.groupby(["date", "cohort", "symbol"])["delta"].sum()
           .unstack(["cohort", "symbol"]).reindex(index).fillna(0.0).cumsum())
    for c in sorted({k for k, _ in pos.columns}):
        g = sum((pos[(c, s)] * closes[s]).abs() for s in pos[c].columns)
        out[f"cohort_{c}"] = g / equity.reindex(index)
    out["total"] = out.sum(axis=1)
    return out


# --------------------------------------------------------------------------- stability
def stability_analysis(plans: list[CohortPlan], close: pd.DataFrame, cfg: ScreenConfig,
                       forward_months: int = 12, data_end: pd.Timestamp | None = None
                       ) -> pd.DataFrame:
    """Per selected pair: does the relationship persist?

    * ``next_selected`` / ``next_fdr``: in the NEXT cohort's screen (formation windows
      overlap heavily - this is persistence, NOT independent forward validation).
    * ``fwd_*``: EG test, beta and half-life on the non-overlapping ``forward_months``
      after the cutoff (only if fully observed before ``data_end``), orientation fixed.
    """
    rows = []
    end = pd.Timestamp(data_end) if data_end is not None else close.index[-1]
    nxt = {i: plans[i + 1] for i in range(len(plans) - 1)}
    for i, pl in enumerate(plans):
        cand_next = nxt[i].screen.candidates if i in nxt else pd.DataFrame()
        for r in pl.screen.selected.to_dict("records"):
            row = {"cohort": pl.cohort.cohort_id, "cutoff": pl.cohort.cutoff, "pair": r["pair"],
                   "beta_form": r["beta"], "hl_form": r.get("ou_half_life", math.nan),
                   "p_form": r["p_pair"]}
            if not cand_next.empty and "pair" in cand_next:
                m = cand_next[cand_next["pair"] == r["pair"]]
                row["next_selected"] = bool(m["selected"].any()) if len(m) else False
                row["next_fdr"] = bool(m["fdr_reject"].any()) if len(m) else False
            f_end = pl.cohort.cutoff + pd.DateOffset(months=forward_months)
            if f_end <= end:
                w = log_prices(close.loc[(close.index > pl.cohort.cutoff) & (close.index <= f_end),
                                     [r["y"], r["x"]]].dropna())
                eg = engle_granger(w[r["y"]], w[r["x"]], trend=cfg.trend, autolag=cfg.autolag)
                row.update({"fwd_p": eg.pvalue, "fwd_reject_5pct": eg.pvalue < 0.05,
                            "fwd_beta": eg.beta, "beta_drift": eg.beta - r["beta"]})
                if eg.status == "ok":
                    ou = fit_ou(w[r["y"]] - eg.alpha - eg.beta * w[r["x"]])
                    row["fwd_half_life"] = ou.half_life if ou.valid else math.nan
                    row["fwd_ou_status"] = ou.status
                    # frozen formation coefficients applied forward (what static OLS trades)
                    s_frozen = w[r["y"]] - r["alpha"] - r["beta"] * w[r["x"]]
                    ou2 = fit_ou(s_frozen)
                    row["fwd_frozen_ou_status"] = ou2.status
                    row["fwd_frozen_half_life"] = ou2.half_life if ou2.valid else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def oos_spread_adf(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Plain ADF on trading-window spreads per hedge method.

    NOTE: for a FIXED (static) vector this is an ordinary unit-root test.  For rolling
    and Kalman spreads the coefficients were re-estimated along the way, so the ADF
    p-value has no standard interpretation (adaptive residuals look more stationary).
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for k, s in series.items():
        s = s.dropna()
        if len(s) < 30 or s.std() < 1e-12:
            rows.append({"series": k, "n": len(s)})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            st, p, *_ = adfuller(s.to_numpy(), regression="c", autolag="AIC")
        rows.append({"series": k, "n": len(s), "adf_stat": float(st), "adf_p_nominal": float(p)})
    return pd.DataFrame(rows)


def cohort_table(cohorts: list[Cohort], plans: list[CohortPlan]) -> pd.DataFrame:
    rows = []
    for c, pl in zip(cohorts, plans, strict=True):
        rows.append({**c.as_dict(), **pl.screen.funnel, "traded_pairs": len(pl.pairs),
                     "pairs": ",".join(p.pair for p in pl.pairs)})
    return pd.DataFrame(rows)
