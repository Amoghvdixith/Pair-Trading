"""Experiment orchestration: every chart/table in results/ is produced here.

Sample labels used in outputs
-----------------------------
* TRAINING / FORMATION - screening and estimation windows inside each cohort
* VALIDATION           - trading windows 2013-2016 (synthetic) used ONLY for tuning
* OOS (a-priori)       - test trading windows with parameters fixed in configs/ before
                         any test result was seen (headline)
* OOS (validation-selected) - one evaluation of the validation-chosen setting
* EXPLORATORY          - the full grid evaluated on the test period; NOT used to choose
* HOLD-OUT             - physically removed; never evaluated by run_all
"""

from __future__ import annotations

import importlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from backtester import BacktestResult, MarketData
from backtester.walkforward import TrialRegistry, check_holdout_lock
from scipy import stats

from . import analysis as A
from . import data as D
from . import plots as P
from .backtester_adapter import (
    RunSpec,
    backtester_version,
    market_data_from_long,
    run_pairs_backtest,
)
from .cohorts import Cohort, make_schedule
from .hedge import kalman_filter, ols_hedge, pykalman_reference, rolling_ols_hedge, tune_kalman
from .ou import fit_ou, variance_ratio_test
from .screening import (
    ScreenConfig,
    benjamini_hochberg,
    benjamini_yekutieli,
    engle_granger,
    johansen,
    unit_root_diagnostics,
)
from .signals import rolling_zscore
from .strategy import CohortPlan, PairsStrategy, StrategyParams, build_plans, sectors_map


def project_root(start: str | Path | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists() and (q / "configs").is_dir():
            return q
    raise FileNotFoundError("project root (pyproject.toml + configs/) not found")


def load_yaml(p: Path) -> dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- context
@dataclass
class Context:
    root: Path
    data_mode: str
    mode: str
    exp: dict[str, Any]
    screen_raw: dict[str, Any]
    strat: dict[str, Any]
    cohort_cfg: dict[str, Any]
    dataset: str = ""
    ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    bench: pd.Series | None = None
    pair_truth: pd.DataFrame | None = None
    data_meta: dict[str, Any] = field(default_factory=dict)
    entries: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    _plans: dict[str, list[CohortPlan]] = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        return "syn" if self.data_mode == "synthetic" else "mkt"

    @property
    def fig(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def tab(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def logs(self) -> Path:
        return self.root / "results" / "logs"

    @property
    def screen_cfg(self) -> ScreenConfig:
        return ScreenConfig.from_dict(self.screen_raw)

    @property
    def split(self) -> dict[str, Any]:
        return self.exp["splits"][self.data_mode]

    @property
    def close(self) -> pd.DataFrame:
        if "_close" not in self.__dict__:
            self.__dict__["_close"] = D.wide(self.ohlcv, "close")
            self.__dict__["_volume"] = D.wide(self.ohlcv, "volume")
        return self.__dict__["_close"]

    @property
    def volume(self) -> pd.DataFrame:
        _ = self.close
        return self.__dict__["_volume"]

    def cohorts(self, which: str) -> list[Cohort]:
        lo, hi = self.split[f"{which}_trading"]
        step = int(self.cohort_cfg["step_months"]) if self.mode == "fast" else 1
        return make_schedule(self.close.index, int(self.cohort_cfg["formation_months"]),
                             int(self.cohort_cfg["trading_months"]), step,
                             first_trade_month=lo[:7], last_trade_end=hi)

    def plans(self, which: str) -> list[CohortPlan]:
        if which not in self._plans:
            t0 = time.perf_counter()
            self._plans[which] = build_plans(self.close, self.volume, self.meta,
                                             self.cohorts(which), self.screen_cfg, self.strat)
            self.timings[f"plans_{which}"] = time.perf_counter() - t0
        return self._plans[which]

    def market_data(self, which: str) -> MarketData:
        cache = self.__dict__.setdefault("_md", {})
        if which in cache:
            return cache[which]
        cache[which] = self._build_md(which)
        return cache[which]

    def _build_md(self, which: str) -> MarketData:
        coh = self.cohorts(which)
        lo, hi = coh[0].formation_start, coh[-1].trade_end
        df = self.ohlcv[(self.ohlcv["date"] >= lo) & (self.ohlcv["date"] <= hi)]
        return market_data_from_long(df, sorted(self.meta["symbol"]),
                                     {"data_mode": self.data_mode, "dataset": self.dataset})

    def stamp(self, which: str, **kw: str) -> P.Stamp:
        coh = self.cohorts(which) if which in ("validation", "test") else []
        dates = (f"{coh[0].trade_start.date()} to {coh[-1].trade_end.date()} (trading)"
                 if coh else f"{self.ohlcv['date'].min().date()} to "
                             f"{self.ohlcv['date'].max().date()}")
        step = self.cohort_cfg["step_months"] if self.mode == "fast" else 1
        base = P.Stamp(self.data_mode, f"{self.dataset} ({self.meta.shape[0]} stocks, "
                                       f"same-industry pairs)", dates,
                       design=f"formation {self.cohort_cfg['formation_months']}m / trading "
                              f"{self.cohort_cfg['trading_months']}m, new cohort every "
                              f"{step}m", costs=cost_label(self.strat),
                       sample={"validation": "VALIDATION", "test": "OOS (a-priori params)"}
                       .get(which, which))
        return base.with_(**kw)


def cost_label(strat: dict[str, Any]) -> str:
    c, f = strat["costs"], strat["financing"]
    return (f"commission {c['commission_bps']}bp + half of {c['spread_bps_full']}bp spread + "
            f"{c['slippage_bps']}bp slippage per leg; borrow {f['borrow_fee']:.1%}/yr; "
            f"cash {f['credit_rate']:.0%}, debit {f['debit_rate']:.0%}")


def build_context(root: Path, data_mode: str = "synthetic", mode: str = "fast") -> Context:
    if data_mode not in ("synthetic", "market"):
        raise ValueError("data_mode must be synthetic or market")
    if mode not in ("fast", "research"):
        raise ValueError("mode must be fast or research")
    cfg = root / "configs"
    ctx = Context(root, data_mode, mode, load_yaml(cfg / "experiments.yaml"),
                  load_yaml(cfg / "screening.yaml"), load_yaml(cfg / "strategy.yaml"),
                  load_yaml(cfg / "cohorts.yaml"))
    if data_mode == "synthetic":
        ctx.dataset = f"synthetic_{mode}"
        d = root / "data" / "synthetic"
        if not (d / f"{ctx.dataset}.csv").exists():
            generate_synthetic(root, mode)
        ds = D.load_dataset(d, ctx.dataset)
    else:
        ctx.dataset = "market_us"
        d = root / "data" / "market"
        if not (d / f"{ctx.dataset}.csv").exists():
            raise FileNotFoundError(
                "market data not cached: run scripts/download_market_data.py first "
                "(no synthetic substitution is made)")
        ds = D.load_dataset(d, ctx.dataset)
    hs = pd.Timestamp(ctx.split["holdout_start"])
    check_holdout_lock(ctx.logs / "holdout_lock.json", hs, ctx.dataset)
    oh = ds["ohlcv"]
    ctx.ohlcv = oh[oh["date"] < hs].reset_index(drop=True)  # hold-out physically removed
    ctx.meta = ds["universe"]
    ctx.bench = ds["benchmark"][ds["benchmark"].index < hs] if ds["benchmark"] is not None else None
    ctx.pair_truth = ds.get("pair_truth")
    ctx.data_meta = ds["meta"] | {"holdout_start_excluded": str(hs.date())}
    return ctx


def generate_synthetic(root: Path, mode: str = "fast") -> dict[str, str]:
    exp = load_yaml(root / "configs" / "experiments.yaml")
    s = exp["synthetic"]
    spec = D.UniverseSpec(start=s["start"], end=s["end"], seed=int(exp["seed"]), **s[mode])
    return D.write_synthetic_dataset(spec, root / "data" / "synthetic", f"synthetic_{mode}")


# --------------------------------------------------------------------------- helpers
def run(ctx: Context, which: str, method: str = "static_ols", scenario: str = "net",
        flat_bps: float = 0.0, plans: list[CohortPlan] | None = None,
        record: set[str] | None = None, label: str = "", **overrides: Any
        ) -> tuple[BacktestResult, PairsStrategy]:
    plans = plans if plans is not None else ctx.plans(which)
    params = StrategyParams.from_config(ctx.strat, ctx.screen_cfg, method=method, **overrides)
    strat = PairsStrategy(plans, params, sectors_map(ctx.meta), record_pairs=record, label=label)
    md = ctx.market_data(which)
    t0 = time.perf_counter()
    res = run_pairs_backtest(md, strat, ctx.strat, RunSpec(scenario, flat_bps),
                             plans[0].cohort.trade_start if plans else ctx.cohorts(which)[0].trade_start,
                             label or method, ctx.data_mode)
    res.meta["runtime_seconds_total"] = time.perf_counter() - t0
    return res, strat


def summary(ctx: Context, res: BacktestResult, strat: PairsStrategy, scenario: str
            ) -> dict[str, Any]:
    rf = float(ctx.strat["financing"]["credit_rate"]) if scenario != "gross" else 0.0
    m = A.performance(res, rf, benchmark=ctx.bench)
    tr = pd.DataFrame(strat.trades)
    n_closed = int(tr["exit_fill"].notna().sum()) if "exit_fill" in tr else 0
    ev = pd.DataFrame(strat.events)
    out = {k: m.get(k) for k in ("start", "end", "cagr", "ann_vol", "sharpe", "sortino",
                                 "max_drawdown", "turnover_annual", "avg_gross_exposure",
                                 "avg_net_exposure", "time_invested", "commissions",
                                 "spread_costs", "slippage_costs", "total_trading_costs",
                                 "interest", "borrow_fees")}
    out["market_beta_raw"] = m.get("beta")
    # with no trades the excess return is numerical noise around zero: Sharpe undefined
    r = res.returns()
    if rf:
        from backtester.performance import rf_from_annual
        r = r - rf_from_annual(r.index, rf)
    if float(r.std()) * math.sqrt(252) < 1e-6:
        out["sharpe"] = out["sortino"] = math.nan
    ser = pd.DataFrame(strat.series)
    if len(ser) and "z" in ser:
        zz = ser["z"].astype(float).dropna()
        out["share_abs_z_ge_entry"] = float((zz.abs() >= strat.p.signal.entry_z).mean())
        out["z_sd"] = float(zz.std())
    out["completed_pair_trades"] = n_closed
    out["avg_holding_sessions"] = float(tr["holding_sessions"].mean()) if n_closed else math.nan
    out["leg_failures"] = int((ev["event"] == "leg_failure").sum()) if len(ev) else 0
    out["breakdown_exits"] = int(((ev["event"] == "exit_signal") &
                                  (ev.get("reason") == "breakdown")).sum()) if len(ev) else 0
    out["stop_loss_exits"] = int(((ev["event"] == "exit_signal") &
                                  (ev.get("reason") == "stop_loss")).sum()) if len(ev) else 0
    out["rehedges"] = int((ev["event"] == "rehedge_confirmed").sum()) if len(ev) else 0
    out["sharpe_rf"] = rf
    out["backstop_bound"] = res.meta.get("backstop_bound")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --------------------------------------------------------------------------- E1 examples
def synthetic_examples(ctx: Context) -> dict[str, Any]:
    seed = int(ctx.exp["seed"])
    ex = {"A cointegrated": D.cointegrated_pair(2000, seed=seed),
          "B independent RW": D.independent_random_walks(2000, seed=seed),
          "C correlated RW (rho=0.8)": D.correlated_random_walks(2000, 0.8, seed=seed),
          "D breakdown at t=1200": D.breaking_pair(2000, 1200, seed=seed)}
    rows = []
    fig, axes = P.plt.subplots(2, 4, figsize=(15, 6.2))
    fdata = {}
    for j, (name, (df, truth)) in enumerate(ex.items()):
        eg = engle_granger(df["y"], df["x"])
        eg_r = engle_granger(df["x"], df["y"])
        jo = johansen(df["y"], df["x"])
        ur = unit_root_diagnostics(df["y"])
        fit = ols_hedge(df["y"], df["x"])
        ou = fit_ou(fit.resid)
        corr = float(df["y"].diff().corr(df["x"].diff()))
        rows.append({"example": name, "truth": truth.kind, "n": len(df),
                     "return_corr": corr, "beta_hat": fit.beta, "beta_true": truth.beta,
                     "eg_p_y_on_x": eg.pvalue, "eg_p_x_on_y": eg_r.pvalue,
                     "johansen_rank_5pct": jo.get("johansen_rank"),
                     "y_level_unit_root_label": ur.get("label"),
                     "ou_status": ou.status, "half_life_hat": ou.half_life,
                     "half_life_true": truth.half_life})
        ax = axes[0, j]
        ax.plot(df.index, df["y"], label="log y", color=P.SERIES[0])
        ax.plot(df.index, df["x"], label="log x", color=P.SERIES[1])
        ax.set_title(name)
        ax.legend(loc="upper left")
        ax2 = axes[1, j]
        ax2.plot(fit.resid.index, fit.resid, color=P.SERIES[2], lw=0.8)
        ax2.axhline(0, color=P.INK2, lw=0.6)
        ax2.set_title(f"OLS residual  EG p={eg.pvalue:.3g}, return corr={corr:.2f}",
                      fontsize=8.5)
        fdata[name + " y"], fdata[name + " x"] = df["y"], df["x"]
        fdata[name + " resid"] = fit.resid
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tab / "syn_01_examples_tests.csv", index=False)
    P.finish(fig, ctx.fig / "syn_01_cointegrated_vs_not.png",
             P.Stamp("synthetic", "two-series examples A-D (n=2000, log prices)",
                     "2000 business days", "single sample, no trading", "OLS",
                     "n/a", "n/a", "STATISTICAL DEMONSTRATION"),
             pd.DataFrame(fdata))
    # Kalman / time-varying beta example (E)
    df, _ = D.time_varying_beta_pair(2000, seed=seed)
    form = df.iloc[:500]
    tune = tune_kalman(form["y"], form["x"], [1e-7, 1e-6, 1e-5, 1e-4])
    kf, _ = kalman_filter(df["y"], df["x"], tune.Q, tune.R, np.array(tune.m0),
                          np.diag(tune.P0_diag))
    ro = rolling_ols_hedge(df["y"], df["x"], 120)
    st = ols_hedge(form["y"], form["x"])
    ref = pykalman_reference(df["y"], df["x"], tune.Q, tune.R, np.array(tune.m0),
                             np.diag(tune.P0_diag))
    ref_diff = (float(np.abs(ref.to_numpy() - kf[["alpha", "beta"]].to_numpy()).max())
                if ref is not None else math.nan)
    fig, ax = P.plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(df.index, df["beta_true"], color=P.INK, lw=2, label="true beta_t")
    ax[0].axhline(st.beta, color=P.SERIES[0], label=f"static OLS (first 500 obs) {st.beta:.2f}")
    ax[0].plot(ro.index, ro["beta"], color=P.SERIES[1], label="rolling OLS (120, t-1)")
    ax[0].plot(kf.index, kf["beta"], color=P.SERIES[2],
               label=f"Kalman filter (delta={tune.delta:g}, tuned on first 500)")
    ax[0].axvline(df.index[500], color=P.INK2, ls=":", lw=0.8)
    ax[0].set_title("Example E: time-varying hedge ratio")
    ax[0].legend(loc="upper left", fontsize=7.5)
    zin = kf["z_innovation"]
    zpost = kf["post_resid"] / math.sqrt(tune.R)
    ax[1].plot(zin.index, zin, lw=0.6, color=P.SERIES[2], label="innovation z (pre-update)")
    ax[1].plot(zpost.index, zpost, lw=0.6, color=P.SERIES[6],
               label="posterior residual / sqrt(R) (uses y_t: NOT tradable)")
    ax[1].set_title(f"Innovation vs posterior residual (sd {zin.std():.2f} vs {zpost.std():.2f})")
    ax[1].legend(loc="upper left", fontsize=7.5)
    P.finish(fig, ctx.fig / "syn_05b_kalman_time_varying_beta.png",
             P.Stamp("synthetic", "example E (n=2000)", "2000 business days",
                     "tuning on first 500 obs, filter thereafter", "static/rolling/Kalman",
                     "n/a", "n/a", "STATISTICAL DEMONSTRATION"),
             pd.DataFrame({"beta_true": df["beta_true"], "rolling": ro["beta"],
                           "kalman": kf["beta"], "z_innov": zin, "z_post": zpost}))
    kal = {"delta": tune.delta, "R": tune.R, "loglik_grid": tune.grid,
           "pykalman_max_abs_diff": ref_diff,
           "pykalman_available": ref is not None,
           "kalman_beta_rmse_after_500": float(np.sqrt(((kf["beta"] - df["beta_true"])
                                                        .iloc[500:] ** 2).mean())),
           "rolling_beta_rmse_after_500": float(np.sqrt(((ro["beta"] - df["beta_true"])
                                                         .iloc[500:] ** 2).mean())),
           "static_beta_rmse_after_500": float(np.sqrt(((st.beta - df["beta_true"])
                                                        .iloc[500:] ** 2).mean()))}
    # initialisation sensitivity
    sens = []
    for pv in (0.01, 1.0, 100.0):
        k2, _ = kalman_filter(df["y"], df["x"], tune.Q, tune.R, np.array([0.0, 1.0]),
                              np.diag([pv, pv]))
        sens.append({"prior": f"m0=[0,1], P0={pv}*I", "beta_t50": float(k2["beta"].iloc[50]),
                     "beta_t500": float(k2["beta"].iloc[500]),
                     "beta_end": float(k2["beta"].iloc[-1])})
    pd.DataFrame(sens).to_csv(ctx.tab / "syn_05b_kalman_init_sensitivity.csv", index=False)
    out = {"examples": t.to_dict("records"), "kalman_example": kal}
    (ctx.tab / "syn_05b_kalman_example.json").write_text(json.dumps(out["kalman_example"],
                                                                    indent=2, default=str))
    return out


def calibration(ctx: Context) -> pd.DataFrame:
    """Null-rejection and recovery simulations (statistical calibration, not unit tests)."""
    n_rep = 300 if ctx.mode == "fast" else 1000
    base = int(ctx.exp["seed"]) * 10
    rows = []

    def add(name: str, null: str, k: int, n: int, extra: dict[str, Any] | None = None) -> None:
        lo, hi = wilson(k, n)
        rows.append({"experiment": name, "null_or_truth": null, "replications": n,
                     "rejections": k, "rate": k / n if n else math.nan, "wilson95_low": lo,
                     "wilson95_high": hi, **(extra or {})})

    for nobs in (252, 1000):
        k = sum(engle_granger(*(lambda d: (d["y"], d["x"]))(
            D.independent_random_walks(nobs, seed=base + i)[0])).pvalue < 0.05
            for i in range(n_rep))
        add(f"EG 5% test, independent RWs, n={nobs}", "H0 true (no cointegration)", k, n_rep)
    k = sum(engle_granger(*(lambda d: (d["y"], d["x"]))(
        D.correlated_random_walks(252, 0.8, seed=base + i)[0])).pvalue < 0.05
        for i in range(n_rep))
    add("EG 5% test, correlated RWs rho=0.8, n=252", "H0 true", k, n_rep)
    for hl in (5.0, 10.0, 30.0):
        k = sum(engle_granger(*(lambda d: (d["y"], d["x"]))(
            D.cointegrated_pair(252, half_life=hl, noise_sd=0.03, seed=base + i)[0])).pvalue
            < 0.05 for i in range(n_rep))
        add(f"EG 5% power, cointegrated hl={hl:g}, n=252", "H1 true", k, n_rep)
    # near-cointegrated null: y = x + AR(1) noise + SMALL random walk (no cointegration)
    for idio in (0.003, 0.009):
        k = 0
        for i in range(n_rep):
            g = np.random.default_rng(base + 50_000 + i)
            x = D.random_walk(252, 0.013, g)
            y = x + D.simulate_ar1(252, 10.0, 0.03, g) + D.random_walk(252, idio, g)
            k += engle_granger(pd.Series(y), pd.Series(x)).pvalue < 0.05
        add(f"EG under near-cointegrated null: AR(1) hl=10 sd=0.03 + RW sd={idio}, n=252",
            "H0 true (residual has a unit root)", k, n_rep)
    ranks = np.array([johansen(*(lambda d: (d["y"], d["x"]))(
        D.independent_random_walks(252, seed=base + i)[0])).get("johansen_rank", -1)
        for i in range(n_rep)])
    add("Johansen trace 5%, det_order=0, k=1: rank>=1, independent RWs n=252",
        "H0 true (rank 0)", int((ranks >= 1).sum()), n_rep,
        {"share_rank2": float((ranks == 2).mean())})
    betas, hls = [], []
    for i in range(min(n_rep, 300)):
        df, _ = D.cointegrated_pair(2000, seed=base + i)
        betas.append(ols_hedge(df["y"], df["x"]).beta)
        hls.append(fit_ou(df["u"]).half_life)
    b, h = np.array(betas), np.array(hls)
    rows.append({"experiment": "beta recovery (true 1.3), n=2000", "null_or_truth": "beta=1.3",
                 "replications": len(b), "mean": b.mean(), "bias": b.mean() - 1.3, "sd": b.std(),
                 "share_within_0.05": float((np.abs(b - 1.3) < 0.05).mean())})
    rows.append({"experiment": "OU half-life recovery (true 10), n=2000, true noise",
                 "null_or_truth": "half-life=10", "replications": len(h), "mean": h.mean(),
                 "median": float(np.median(h)), "bias": h.mean() - 10, "sd": h.std(),
                 "share_within_15pct": float((np.abs(h / 10 - 1) < 0.15).mean())})
    rng = np.random.default_rng(base)
    k_h = k_o = 0
    for _i in range(n_rep):
        X = np.cumsum(rng.standard_t(5, 1000) * 0.01)
        vr = variance_ratio_test(X, 5)
        k_h += vr["p_het"] < 0.05
        k_o += vr["p_hom"] < 0.05
    add("Variance ratio q=5 robust z, RW t(5) increments n=1000", "H0 true", k_h, n_rep)
    add("Variance ratio q=5 homoskedastic z, same data", "H0 true", k_o, n_rep)
    # BH FDR: independent tests, 40 nulls + 10 alternatives
    fdps, powers = [], []
    for _i in range(n_rep):
        z = np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 10)])
        p = stats.norm.sf(z)
        rej, _ = benjamini_hochberg(p, 0.10)
        v = rej[:40].sum()
        fdps.append(v / max(rej.sum(), 1))
        powers.append(rej[40:].mean())
    rows.append({"experiment": "BH q=0.10, 50 independent one-sided z-tests (10 alt)",
                 "null_or_truth": "FDR <= q*m0/m = 0.08", "replications": n_rep,
                 "mean": float(np.mean(fdps)), "sd": float(np.std(fdps)),
                 "power": float(np.mean(powers))})
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tab / "syn_00_calibration.csv", index=False)
    return t


# --------------------------------------------------------------------------- E2 screening
def screening(ctx: Context) -> dict[str, Any]:
    pre = ctx.prefix
    plans = ctx.plans("validation") + ctx.plans("test")
    cand = pd.concat([pl.screen.candidates.assign(cohort=pl.cohort.cohort_id,
                                                  sample=s)
                      for s in ("validation", "test") for pl in ctx.plans(s)],
                     ignore_index=True)
    cand.to_csv(ctx.tab / f"{pre}_screening_audit_all_cohorts.csv", index=False)
    uni = pd.concat([pl.screen.universe_audit.assign(cutoff=pl.cohort.cutoff) for pl in plans])
    uni[~uni["eligible"]].to_csv(ctx.tab / f"{pre}_universe_rejections.csv", index=False)
    funnel = pd.DataFrame([{"cutoff": pl.cohort.cutoff.date(), **pl.screen.funnel,
                            "traded": len(pl.pairs)} for pl in plans])
    funnel.to_csv(ctx.tab / f"{pre}_02_screening_funnel_by_cohort.csv", index=False)
    stages = ["candidates", "raw_reject", "fdr_reject", "passes_filters", "selected"]
    tot = funnel[stages].sum()
    fig, ax = P.plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ax[0].barh(range(len(stages))[::-1], tot.values, color=P.SERIES[0], height=0.6)
    ax[0].set_yticks(range(len(stages))[::-1],
                     ["candidates tested", "raw EG p<0.05", f"{ctx.screen_cfg.fdr_method.upper()}"
                      f" q={ctx.screen_cfg.fdr_q}", "+ OU/beta/crossing filters",
                      "selected (1 pair/stock)"])
    for i, v in enumerate(tot.values[::-1]):
        ax[0].text(v, i, f" {int(v)}", va="center", color=P.INK)
    ax[0].set_title(f"Screening funnel, summed over {len(plans)} cohorts")
    ax[0].grid(axis="y", visible=False)
    x = pd.to_datetime(funnel["cutoff"])
    for i, s in enumerate(stages[1:]):
        ax[1].plot(x, funnel[s], marker="o", ms=3, color=P.SERIES[i], label=s)
    ax[1].set_title("Per formation date (overlapping 12m windows)")
    ax[1].legend(ncol=2, loc="upper left")
    P.finish(fig, ctx.fig / f"{pre}_02_screening_funnel.png",
             ctx.stamp("screen", sample="FORMATION windows (validation + test cohorts)"),
             funnel)
    # candidate table for the first test cohort
    first = ctx.plans("test")[0]
    cols = ["pair", "y", "x", "sector", "eg_stat", "eg_pvalue_yx", "eg_pvalue_xy", "p_pair",
            "p_adj", "raw_reject", "fdr_reject", "beta", "ou_half_life", "zero_crossings",
            "hurst", "vr5", "vr5_p_het", "johansen_rank", "beta_johansen", "return_corr",
            "filter_reasons", "selected"]
    ct = first.screen.candidates[[c for c in cols if c in first.screen.candidates]]
    ct.sort_values("p_pair").to_csv(ctx.tab / f"{pre}_03_candidates_first_test_cohort.csv",
                                    index=False)
    out: dict[str, Any] = {"n_cohorts": len(plans), "funnel_totals": tot.to_dict(),
                           "cohorts_with_no_selection": int((funnel["selected"] == 0).sum()),
                           "first_test_cutoff": str(first.cohort.cutoff.date())}
    ok = cand[cand["eg_status"] == "ok"]
    if "johansen_rank" in ok:
        out["eg_fdr_vs_johansen"] = pd.crosstab(ok["fdr_reject"], ok["johansen_rank"]).to_dict()
        pd.crosstab(ok["fdr_reject"], ok["johansen_rank"]).to_csv(
            ctx.tab / f"{pre}_eg_vs_johansen_crosstab.csv")
    if ctx.pair_truth is not None:
        tr = ctx.pair_truth.set_index("pair")
        rows = []
        for pl in plans:
            c = pl.screen.candidates
            if c.empty:
                continue
            until = pd.to_datetime(tr.loc[c["pair"], "cointegrated_until"]).to_numpy()
            true = (tr.loc[c["pair"], "cointegrated"].to_numpy().astype(bool)
                    & ((pd.isna(until)) | (until > np.datetime64(pl.cohort.cutoff))))
            r = {"cutoff": pl.cohort.cutoff.date(), "m": len(c), "m_true": int(true.sum())}
            by_rej, _ = benjamini_yekutieli(c["p_pair"].to_numpy(), ctx.screen_cfg.fdr_q)
            stage_masks = {stg: c[stg].to_numpy().astype(bool)
                           for stg in ("raw_reject", "fdr_reject", "selected")}
            stage_masks["by_reject"] = by_rej
            for stg, rej in stage_masks.items():
                r[f"{stg}_n"] = int(rej.sum())
                r[f"{stg}_false"] = int((rej & ~true).sum())
                r[f"{stg}_fdp"] = (rej & ~true).sum() / rej.sum() if rej.sum() else 0.0
                r[f"{stg}_power"] = (rej & true).sum() / true.sum() if true.sum() else math.nan
            rows.append(r)
        fd = pd.DataFrame(rows)
        fd.to_csv(ctx.tab / "syn_02b_screening_vs_truth.csv", index=False)
        out["truth"] = {k: float(fd[k].mean()) for k in fd.columns if k.endswith(("fdp", "power"))}
        out["truth"]["raw_false_total"] = int(fd["raw_reject_false"].sum())
        out["truth"]["fdr_false_total"] = int(fd["fdr_reject_false"].sum())
        out["truth"]["selected_false_total"] = int(fd["selected_false"].sum())
        out["truth"]["selected_total"] = int(fd["selected_n"].sum())
        out["truth"]["by_false_total"] = int(fd["by_reject_false"].sum())
        out["truth"]["by_total"] = int(fd["by_reject_n"].sum())
        out["truth"]["fdr_total"] = int(fd["fdr_reject_n"].sum())
    return out


# --------------------------------------------------------------------------- E3 single pair
def single_pair(ctx: Context) -> dict[str, Any]:
    pre = ctx.prefix
    plans = ctx.plans("test")
    first = next((pl for pl in plans if pl.pairs), None)
    if first is None:
        return {"status": "no pair selected in any test cohort"}
    spec = first.pairs[0]
    pl1 = [CohortPlan(first.cohort, first.screen, [spec])]
    res, strat = run(ctx, "test", "static_ols", "net", plans=pl1, label="single_pair")
    c = first.cohort
    close = ctx.close
    win = close.loc[c.formation_start:c.trade_end, [spec.y, spec.x]]
    lp = D.log_prices(win)
    spread = lp[spec.y] - spec.alpha - spec.beta * lp[spec.x]
    sg = ctx.strat["signals"]
    L = int(np.clip(round(float(sg["zscore_lookback_mult"]) * spec.half_life),
                    int(sg["zscore_lookback_min"]), int(sg["zscore_lookback_max"])))
    z = rolling_zscore(spread, L)
    fills = res.fills.copy()
    fills["timestamp"] = pd.to_datetime(fills["timestamp"]) if len(fills) else fills.get("timestamp")
    ev = pd.DataFrame(strat.events)
    fig, ax = P.plt.subplots(3, 1, figsize=(12, 8.4), sharex=True)
    norm = win / win.iloc[0]
    ax[0].plot(norm.index, norm[spec.y], label=f"{spec.y} (y)", color=P.SERIES[0])
    ax[0].plot(norm.index, norm[spec.x], label=f"{spec.x} (x)", color=P.SERIES[1])
    ax[0].set_title(f"{spec.pair}: prices rebased to 1 (formation then trading window)")
    ax[0].legend(loc="upper left")
    ax[1].plot(spread.index, spread, color=P.SERIES[2], lw=1)
    sgn = "-" if spec.alpha >= 0 else "+"
    ax[1].set_title(f"log spread  y {sgn} {abs(spec.alpha):.3f} - {spec.beta:.3f} x  "
                    "(formation OLS, frozen)")
    ax[2].plot(z.index, z, color=P.SERIES[6], lw=1, label=f"z (L={L}, incl. current close)")
    for lvl in (2.0, -2.0):
        ax[2].axhline(lvl, color=P.INK2, lw=0.6, ls="--")
    for lvl in (3.5, -3.5):
        ax[2].axhline(lvl, color=P.SERIES[7], lw=0.6, ls=":")
    ax[2].axhline(0, color=P.INK2, lw=0.6)
    if len(fills):
        fy = fills[fills["symbol"] == spec.y]
        zz = z.reindex(fy["timestamp"])
        ax[2].scatter(fy["timestamp"], zz, s=36, color=P.INK, zorder=5,
                      label="actual fills (next open)")
    ax[2].legend(loc="upper left")
    for a in ax:
        a.axvline(c.cutoff, color=P.INK2, ls=":", lw=0.8)
    P.finish(fig, ctx.fig / f"{pre}_04_single_pair_price_spread_z.png",
             ctx.stamp("test", sample=f"formation to {c.cutoff.date()}, trading after (OOS)"),
             pd.DataFrame({"y": win[spec.y], "x": win[spec.x], "spread": spread, "z": z}))
    # OU diagnostics on formation spread
    fs = spread.loc[:c.cutoff]
    ou = fit_ou(fs)
    fig, ax = P.plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(fs.shift(1), fs, s=6, color=P.SERIES[0])
    xs = np.linspace(fs.min(), fs.max(), 10)
    ax[0].plot(xs, ou.a + ou.b * xs, color=P.SERIES[1], label=f"b={ou.b:.3f}")
    ax[0].set_xlabel("s(t-1)")
    ax[0].set_ylabel("s(t)")
    ax[0].legend()
    ax[0].set_title("AR(1) fit on formation spread")
    hlv = [ou.half_life_ci[0], ou.half_life, min(ou.half_life_ci[1], 200)]
    ax[1].errorbar([0], [ou.half_life], yerr=[[hlv[1] - hlv[0]], [hlv[2] - hlv[1]]],
                   fmt="o", color=P.SERIES[0], capsize=6)
    ax[1].axhspan(2, 60, color=P.GRID, alpha=0.5, label="2-60 heuristic range")
    ax[1].set_xticks([])
    ax[1].set_title(f"half-life {ou.half_life:.1f} sessions, 95% CI "
                    f"[{ou.half_life_ci[0]:.1f}, {ou.half_life_ci[1]:.1f}]")
    ax[1].legend()
    ts = spread.loc[c.trade_start:]
    ax[2].plot(fs.index, fs - ou.mu, color=P.SERIES[0], lw=0.8, label="formation")
    ax[2].plot(ts.index, ts - ou.mu, color=P.SERIES[1], lw=0.8, label="trading (frozen coef.)")
    ax[2].axhline(2 * ou.stationary_sd, color=P.INK2, ls="--", lw=0.6)
    ax[2].axhline(-2 * ou.stationary_sd, color=P.INK2, ls="--", lw=0.6,
                  label="+/-2 stationary sd")
    ax[2].legend(loc="upper left")
    ax[2].set_title("Spread minus OU mean")
    P.finish(fig, ctx.fig / f"{pre}_06_ou_fit_half_life.png",
             ctx.stamp("test", sample="FORMATION (fit); trading shown for comparison"),
             pd.DataFrame({"spread": spread}))
    # hedge paths for this pair (formation + trading)
    ro = rolling_ols_hedge(lp[spec.y], lp[spec.x], int(ctx.strat["hedge"]["rolling_window"]))
    kf, _ = kalman_filter(lp[spec.y], lp[spec.x],
                          np.eye(2) * spec.kalman_delta / (1 - spec.kalman_delta),
                          spec.kalman_R, np.array(spec.kalman_m0), np.diag(spec.kalman_P0))
    fig, ax = P.plt.subplots(1, 1, figsize=(12, 4))
    ax.axhline(spec.beta, color=P.SERIES[0], label=f"static OLS (formation) {spec.beta:.3f}")
    ax.plot(ro.index, ro["beta"], color=P.SERIES[1], label="rolling OLS 60d (window ends t-1)")
    ax.plot(kf.index, kf["beta"], color=P.SERIES[2],
            label=f"Kalman filter (delta={spec.kalman_delta:g}, tuned on formation)")
    ax.axvline(c.cutoff, color=P.INK2, ls=":", lw=0.8)
    ax.set_title(f"{spec.pair}: hedge-ratio paths (filter only, no smoothing)")
    ax.legend(loc="upper left")
    P.finish(fig, ctx.fig / f"{pre}_05_hedge_ratio_paths.png",
             ctx.stamp("test", hedge="static / rolling / Kalman"),
             pd.DataFrame({"static": spec.beta, "rolling": ro["beta"], "kalman": kf["beta"]}))
    # entries/exits with actual fills
    fig, ax = P.plt.subplots(1, 1, figsize=(12, 4.2))
    tz = z.loc[c.trade_start:]
    ax.plot(tz.index, tz, color=P.SERIES[6], lw=1, label="z (decision at close)")
    sig = ev[ev["event"].isin(["entry_signal", "exit_signal"])] if len(ev) else ev
    for e in sig.to_dict("records"):
        d = pd.Timestamp(e["date"])
        ax.scatter([d], [z.get(d, np.nan)], marker="^" if e["event"] == "entry_signal" else "v",
                   s=60, color=P.SERIES[0] if e["event"] == "entry_signal" else P.SERIES[1],
                   zorder=5)
    if len(fills):
        for f in fills[fills["symbol"] == spec.y].to_dict("records"):
            ax.axvline(f["timestamp"], color=P.INK2, lw=0.5, alpha=0.6)
    ax.axhline(2, color=P.INK2, ls="--", lw=0.6)
    ax.axhline(-2, color=P.INK2, ls="--", lw=0.6)
    ax.axhline(0, color=P.INK2, lw=0.6)
    ax.set_title("Signals (triangles, at close) and actual fill sessions (grey lines, next open)")
    P.finish(fig, ctx.fig / f"{pre}_07_entries_exits_fills.png", ctx.stamp("test"),
             fills if len(fills) else None)
    rec = res.reconciliation()
    rec.to_csv(ctx.tab / f"{pre}_07_single_pair_fill_reconciliation.csv", index=False)
    s = summary(ctx, res, strat, "net")
    s.update({"pair": spec.pair, "cohort_cutoff": str(c.cutoff.date()), "hedge_beta": spec.beta,
              "half_life": spec.half_life, "gross_frac": spec.gross_frac})
    return s


# --------------------------------------------------------------------------- E4 portfolio
def portfolio(ctx: Context) -> dict[str, Any]:
    pre = ctx.prefix
    st = ctx.stamp("test")
    md = ctx.market_data("test")
    res_n, s_n = run(ctx, "test", "static_ols", "net", record=set(), label="net")
    res_g, s_g = run(ctx, "test", "static_ols", "gross", record=set(), label="gross")
    sn, sg = summary(ctx, res_n, s_n, "net"), summary(ctx, res_g, s_g, "gross")
    pd.DataFrame({"gross": sg, "net": sn}).to_csv(ctx.tab / f"{pre}_10_portfolio_metrics.csv")
    res_n.save(ctx.logs / f"{pre}_portfolio_net", prefix="net")
    eg_, en_ = res_g.equity_curve(), res_n.equity_curve()
    # 08 equity
    fig, ax = P.plt.subplots(1, 1, figsize=(12, 4.2))
    ax.plot(eg_.index, eg_ / eg_.iloc[0], color=P.SERIES[0], label="gross (no costs/financing)")
    ax.plot(en_.index, en_ / en_.iloc[0], color=P.SERIES[1],
            label="net (costs + borrow + cash interest)")
    if ctx.bench is not None:
        b = ctx.bench.reindex(en_.index).ffill()
        ax.plot(b.index, b / b.iloc[0], color=P.INK2, lw=0.8, label="market benchmark")
    ax.set_title("Pairs portfolio equity (static OLS, a-priori thresholds)")
    ax.legend(loc="upper left")
    P.finish(fig, ctx.fig / f"{pre}_08_equity_gross_net.png", st,
             pd.DataFrame({"gross": eg_, "net": en_}))
    # 09 drawdown
    fig, ax = P.plt.subplots(1, 1, figsize=(12, 3.4))
    for e, nm, col in ((eg_, "gross", P.SERIES[0]), (en_, "net", P.SERIES[1])):
        ax.plot(e.index, P.drawdown(e), color=col, label=nm)
    P.pct(ax)
    ax.set_title("Drawdown")
    ax.legend(loc="lower left")
    P.finish(fig, ctx.fig / f"{pre}_09_drawdown.png", st,
             pd.DataFrame({"gross": P.drawdown(eg_), "net": P.drawdown(en_)}))
    # 10 tearsheet
    r = res_n.returns()
    fig, ax = P.plt.subplots(2, 2, figsize=(13, 7))
    ax[0, 0].plot(en_.index, en_ / en_.iloc[0], color=P.SERIES[1])
    ax[0, 0].set_title("Net equity")
    from backtester.performance import rf_from_annual
    xr = r - rf_from_annual(r.index, float(ctx.strat["financing"]["credit_rate"]))
    rs = (xr.rolling(126).mean() / xr.rolling(126).std() * np.sqrt(252))
    ax[0, 1].plot(rs.index, rs, color=P.SERIES[0])
    ax[0, 1].axhline(0, color=P.INK2, lw=0.6)
    ax[0, 1].set_title("Rolling 6-month Sharpe (excess of 2% cash rate)")
    mo = (1 + r).resample("ME").prod() - 1
    ax[1, 0].bar(mo.index, mo.values, width=20,
                 color=[P.SERIES[0] if v >= 0 else P.SERIES[1] for v in mo.values])
    P.pct(ax[1, 0], decimals=1)
    ax[1, 0].set_title("Monthly net returns")
    ax[1, 1].axis("off")
    keys = [("CAGR", "cagr", "{:.2%}"), ("Ann. vol", "ann_vol", "{:.2%}"),
            ("Sharpe (excess of 2%)", "sharpe", "{:.2f}"), ("Sortino", "sortino", "{:.2f}"),
            ("Max drawdown", "max_drawdown", "{:.2%}"),
            ("Turnover (1-way, /yr)", "turnover_annual", "{:.1f}x"),
            ("Avg gross exposure", "avg_gross_exposure", "{:.2f}"),
            ("Avg net exposure", "avg_net_exposure", "{:.3f}"),
            ("Completed pair trades", "completed_pair_trades", "{:.0f}"),
            ("Avg holding (sessions)", "avg_holding_sessions", "{:.1f}"),
            ("Trading costs", "total_trading_costs", "{:,.0f}"),
            ("Borrow fees", "borrow_fees", "{:,.0f}"), ("Cash interest", "interest", "{:,.0f}")]
    txt = "\n".join(f"{k:<26}{(f.format(sn[v]) if sn.get(v) is not None and np.isfinite(sn[v]) else 'n/a'):>14}"
                    f"   gross {(f.format(sg[v]) if sg.get(v) is not None and np.isfinite(sg[v]) else 'n/a')}"
                    for k, v, f in keys)
    ax[1, 1].text(0, 1, txt, family="monospace", fontsize=8, va="top", color=P.INK)
    ax[1, 1].set_title("Net (left) vs gross")
    P.finish(fig, ctx.fig / f"{pre}_10_tearsheet.png", st,
             pd.DataFrame({"net_return": r}))
    # 11 cost decomposition
    eq = res_n.equity
    costs = pd.DataFrame({k: eq[k] - eq[k].iloc[0] for k in
                          ("commissions", "spread_costs", "slippage_costs", "borrow_fees")})
    costs["interest_paid_or_received"] = -(eq["interest"] - eq["interest"].iloc[0])
    fig, ax = P.plt.subplots(1, 2, figsize=(13, 4))
    for i, c in enumerate(costs.columns):
        ax[0].plot(costs.index, costs[c], color=P.SERIES[i], label=c)
    ax[0].set_title("Cumulative costs (negative interest line = interest earned)")
    ax[0].legend(loc="upper left", fontsize=7.5)
    last = costs.iloc[-1]
    ax[1].barh(range(len(last)), last.values, color=P.SERIES[0], height=0.6)
    ax[1].set_yticks(range(len(last)), last.index)
    ax[1].set_title("Totals over the OOS period (currency units)")
    P.finish(fig, ctx.fig / f"{pre}_11_cost_decomposition.png", st, costs)
    # attribution & trades
    pnl, rec = A.attribute_pairs(res_n, s_n, md)
    rec.to_csv(ctx.tab / f"{pre}_21_pair_attribution_reconciliation.csv")
    trades = A.trade_table(s_n, pnl)
    trades.to_csv(ctx.tab / f"{pre}_18_pair_trades.csv", index=False)
    pd.DataFrame(s_n.holdings_audit).to_csv(ctx.tab / f"{pre}_holdings_intended_vs_realised.csv",
                                            index=False)
    pd.DataFrame(s_n.events).to_csv(ctx.logs / f"{pre}_portfolio_net_pair_events.csv",
                                    index=False)
    # 18 distributions
    fig, ax = P.plt.subplots(1, 2, figsize=(12, 4))
    if len(trades):
        ax[0].hist(trades["holding_sessions"], bins=20, color=P.SERIES[0])
        ax[1].hist(trades["return_on_gross"], bins=30, color=P.SERIES[1])
        P.pct(ax[1], "x", 1)
    ax[0].set_title("Holding period (sessions)")
    ax[1].set_title("Pair-trade P&L / gross notional (net of trading costs, excl. financing)")
    P.finish(fig, ctx.fig / f"{pre}_18_holding_and_trade_pnl.png", st,
             trades[["holding_sessions", "return_on_gross"]] if len(trades) else None)
    # 15/16 neutrality
    neut: dict[str, Any] = {}
    if ctx.bench is not None:
        mret = ctx.bench.pct_change()
        rf = None
        from backtester.performance import rf_from_annual
        rf = rf_from_annual(r.index, float(ctx.strat["financing"]["credit_rate"]))
        neut["net_excess_daily"] = A.neutrality_regression(r, mret, rf=rf)
        neut["gross_daily"] = A.neutrality_regression(res_g.returns(), mret)
        wk = (1 + r).resample("W-FRI").prod() - 1
        wm = (1 + mret.reindex(r.index)).resample("W-FRI").prod() - 1
        neut["net_weekly_raw"] = A.neutrality_regression(wk, wm, hac_lags=4)
        nt = pd.DataFrame(neut).T
        nt.to_csv(ctx.tab / f"{pre}_15_neutrality_regression.csv")
        rb = A.rolling_beta(r, mret, 126)
        fig, ax = P.plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(rb.index, rb, color=P.SERIES[0], label="rolling 126d beta to market")
        ax[0].axhline(0, color=P.INK2, lw=0.6)
        ax[0].legend(loc="upper left")
        ax[0].set_title("Market beta is estimated, not guaranteed")
        ex = res_n.equity.loc[r.index, ["gross_exposure", "net_exposure"]]
        ax[1].plot(ex.index, ex["gross_exposure"], color=P.SERIES[1], label="gross exposure")
        ax[1].plot(ex.index, ex["net_exposure"], color=P.SERIES[2], label="net (dollar) exposure")
        ax[1].legend(loc="upper left")
        ax[1].set_title("Exposure as fraction of equity")
        P.finish(fig, ctx.fig / f"{pre}_16_rolling_beta_exposure.png", st,
                 pd.concat([rb, ex], axis=1))
    # 19 cohort audit
    ce = A.cohort_exposure(s_n, md, r.index, res_n.equity["equity"])
    ce.to_csv(ctx.tab / f"{pre}_19_cohort_exposure.csv")
    cap = float(ctx.strat["constraints"]["portfolio_gross_cap"])
    fig, ax = P.plt.subplots(1, 1, figsize=(12, 4))
    cols = [c for c in ce.columns if c.startswith("cohort_")]
    if cols:
        ax.stackplot(ce.index, *(ce[c].fillna(0).to_numpy() for c in cols),
                     colors=[P.SERIES[i % 8] for i in range(len(cols))], alpha=0.85)
    ax.axhline(cap, color=P.INK, ls="--", lw=0.8, label=f"portfolio gross cap {cap}")
    ax.axhline(cap * ctx.cohorts("test")[0].capital_share, color=P.INK2, ls=":", lw=0.8,
               label="per-cohort budget")
    ax.legend(loc="upper left")
    ax.set_title("Gross exposure by cohort (one shared account; colours cycle by cohort id)")
    P.finish(fig, ctx.fig / f"{pre}_19_cohort_exposure_audit.png", st, ce)
    ct = A.cohort_table(ctx.cohorts("test"), ctx.plans("test"))
    ct.to_csv(ctx.tab / f"{pre}_19_cohort_table.csv", index=False)
    return {"net": sn, "gross": sg, "neutrality": neut,
            "attribution_max_abs_unattributed": float(rec["unattributed"].abs().max()),
            "max_total_cohort_gross": float(ce["total"].max()) if len(ce) else 0.0,
            "max_engine_gross": float(res_n.equity.loc[r.index, "gross_exposure"].max()),
            "n_trades_table": int(len(trades)),
            "trade_hit_rate": float((trades["pnl"] > 0).mean()) if len(trades) else math.nan,
            "exit_reasons": trades["exit_reason"].value_counts().to_dict() if len(trades) else {},
            "holdings_max_dollar_imbalance_beta_rule": float(pd.DataFrame(
                s_n.holdings_audit)["dollar_imbalance"].max()) if s_n.holdings_audit else None}


# --------------------------------------------------------------------------- E5 hedge methods
def hedge_comparison(ctx: Context) -> pd.DataFrame:
    pre = ctx.prefix
    rows = []
    adf_series: dict[str, pd.Series] = {}
    for m in ("static_ols", "rolling_ols", "kalman"):
        for sc in ("net", "gross"):
            res, s = run(ctx, "test", m, sc, label=f"{m}_{sc}")
            row = summary(ctx, res, s, sc)
            row.update({"method": m, "scenario": sc})
            rows.append(row)
            if sc == "net":
                ser = pd.DataFrame(s.series)
                if len(ser):
                    for k, g in ser.groupby("key"):
                        adf_series[f"{m}|{k}"] = g.set_index("date")["spread"]
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tab / f"{pre}_14_hedge_method_comparison.csv", index=False)
    adf = A.oos_spread_adf(adf_series)
    if len(adf):
        adf["method"] = adf["series"].str.split("|").str[0]
        adf.to_csv(ctx.tab / f"{pre}_14b_oos_spread_adf_by_method.csv", index=False)
    net = t[t["scenario"] == "net"].set_index("method")
    fig, ax = P.plt.subplots(1, 4, figsize=(15, 3.8))
    for a, (col, ttl) in zip(ax, [("sharpe", "Net Sharpe (excess of 2%)"), ("cagr", "Net CAGR"),
                                  ("turnover_annual", "Turnover (1-way, x/yr)"),
                                  ("total_trading_costs", "Trading costs")], strict=True):
        a.bar(range(3), net[col].astype(float), color=P.SERIES[:3], width=0.6)
        a.set_xticks(range(3), ["static", "rolling", "Kalman"])
        a.set_title(ttl)
        for i, v in enumerate(net[col].astype(float)):
            a.text(i, v, f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}", ha="center",
                   va="bottom", fontsize=8, color=P.INK)
    P.finish(fig, ctx.fig / f"{pre}_14_hedge_method_comparison.png",
             ctx.stamp("test", hedge="static vs rolling 60d vs Kalman"), t)
    return t


# --------------------------------------------------------------------------- E6 costs
def cost_sensitivity(ctx: Context) -> pd.DataFrame:
    pre = ctx.prefix
    rows = []
    for bps in ctx.exp["cost_scenarios_bps"]:
        res, s = run(ctx, "test", "static_ols", "flat", float(bps), label=f"flat{bps}")
        row = summary(ctx, res, s, "flat")
        row["one_way_bps_per_leg"] = bps
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(ctx.tab / f"{pre}_12_cost_sensitivity.csv", index=False)
    fig, ax = P.plt.subplots(1, 3, figsize=(13, 3.6))
    for a, (col, ttl) in zip(ax, [("sharpe", "Sharpe (excess of 2%)"), ("cagr", "CAGR"),
                                  ("max_drawdown", "Max drawdown")], strict=True):
        a.plot(t["one_way_bps_per_leg"], t[col].astype(float), marker="o", color=P.SERIES[0])
        a.set_xlabel("flat one-way cost per leg (bps)")
        a.set_title(ttl)
    P.finish(fig, ctx.fig / f"{pre}_12_cost_sensitivity.png",
             ctx.stamp("test", costs="flat bps per leg REPLACING commission/spread/slippage; "
                                     "borrow & financing unchanged"), t)
    return t


# --------------------------------------------------------------------------- E7 thresholds
def threshold_sensitivity(ctx: Context) -> dict[str, Any]:
    pre = ctx.prefix
    g = ctx.exp["grids"]
    reg = TrialRegistry(ctx.logs / "trial_registry.jsonl")
    stamp_t = datetime.now(UTC).isoformat(timespec="seconds")
    trials = []

    def trial(which: str, status: str, **kw: Any) -> dict[str, Any]:
        res, s = run(ctx, which, "static_ols", "net", label="grid", **kw)
        sm_ = summary(ctx, res, s, "net")
        rec = {"run_utc": stamp_t, "dataset": ctx.dataset, "sample": which,
               "status": status, **kw, "sharpe": sm_["sharpe"], "cagr": sm_["cagr"],
               "max_drawdown": sm_["max_drawdown"], "trades": sm_["completed_pair_trades"],
               "turnover": sm_["turnover_annual"]}
        trials.append(rec)
        return rec

    for e in g["entry_z"]:
        for x in g["exit_z"]:
            trial("validation", "VALIDATION grid", entry_z=float(e), exit_z=float(x))
    val = pd.DataFrame(trials)
    heat_v = val.pivot(index="entry_z", columns="exit_z", values="sharpe")
    best = val.sort_values(["sharpe", "entry_z"], ascending=[False, True]).iloc[0]
    for lm in g["lookback_mult"]:
        trial("validation", "VALIDATION lookback", entry_z=float(best["entry_z"]),
              exit_z=float(best["exit_z"]), lookback_mult=float(lm))
    lb = pd.DataFrame([t for t in trials if t["status"] == "VALIDATION lookback"])
    best_lb = lb.sort_values(["sharpe", "lookback_mult"], ascending=[False, True]).iloc[0]
    chosen = {"entry_z": float(best["entry_z"]), "exit_z": float(best["exit_z"]),
              "lookback_mult": float(best_lb["lookback_mult"])}
    oos_sel = trial("test", "OOS validation-selected (single evaluation)", **chosen)
    for e in g["entry_z"]:
        for x in g["exit_z"]:
            trial("test", "EXPLORATORY outer-test grid (not used for selection)",
                  entry_z=float(e), exit_z=float(x))
    allt = pd.DataFrame(trials)
    reg.append(allt)
    allt.to_csv(ctx.tab / f"{pre}_13_trial_log.csv", index=False)
    exp_ = allt[allt["status"].str.startswith("EXPLORATORY")]
    heat_t = exp_.pivot(index="entry_z", columns="exit_z", values="sharpe")
    for h, nm, ttl, smp in ((heat_v, "13_threshold_heatmap_validation",
                             "VALIDATION Sharpe (used to choose)", "VALIDATION"),
                            (heat_t, "13b_threshold_heatmap_EXPLORATORY_test",
                             "EXPLORATORY outer-test Sharpe (NOT used to choose)",
                             "EXPLORATORY outer test")):
        fig, ax = P.plt.subplots(1, 1, figsize=(6.5, 4.6))
        P.heatmap(ax, h, title=ttl)
        P.finish(fig, ctx.fig / f"{pre}_{nm}.png",
                 ctx.stamp("validation" if smp == "VALIDATION" else "test", sample=smp), h)
    return {"chosen_on_validation": chosen, "validation_best_sharpe": float(best["sharpe"]),
            "oos_validation_selected": oos_sel, "n_trials_logged": int(len(allt)),
            "lookback_validation": lb[["lookback_mult", "sharpe"]].to_dict("records")}


# --------------------------------------------------------------------------- E8 stability
def stability(ctx: Context, portfolio_events: pd.DataFrame | None = None) -> dict[str, Any]:
    pre = ctx.prefix
    plans = ctx.plans("validation") + ctx.plans("test")
    t = A.stability_analysis(plans, ctx.close, ctx.screen_cfg, 12,
                             data_end=ctx.ohlcv["date"].max())
    t.to_csv(ctx.tab / f"{pre}_17_pair_stability.csv", index=False)
    if t.empty:
        return {"status": "no selected pairs"}
    g = t.groupby("cutoff").agg(
        n=("pair", "size"), next_selected=("next_selected", "mean"),
        fwd_reject=("fwd_reject_5pct", "mean"),
        beta_drift_abs=("beta_drift", lambda s: s.abs().median()))
    fig, ax = P.plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(g.index, g["next_selected"], marker="o", ms=3, color=P.SERIES[0],
               label="re-selected in next cohort (OVERLAPPING windows: persistence)")
    ax[0].plot(g.index, g["fwd_reject"], marker="o", ms=3, color=P.SERIES[1],
               label="EG p<0.05 on next NON-overlapping 12m (forward)")
    P.pct(ax[0])
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(loc="lower left")
    ax[0].set_title("Selected-pair stability by formation date")
    ax[1].plot(g.index, g["beta_drift_abs"], marker="o", ms=3, color=P.SERIES[2],
               label="median |forward beta - formation beta|")
    ax[1].legend(loc="upper left")
    P.finish(fig, ctx.fig / f"{pre}_17_pair_stability.png",
             ctx.stamp("screen", sample="formation vs following windows"), g)
    return {"pairs_evaluated": int(len(t)),
            "share_reselected_next_cohort": float(t["next_selected"].mean()),
            "share_forward_eg_reject_5pct": float(t["fwd_reject_5pct"].mean()),
            "median_abs_beta_drift": float(t["beta_drift"].abs().median()),
            "median_half_life_form": float(t["hl_form"].median()),
            "median_half_life_fwd": float(t["fwd_half_life"].median()),
            "share_forward_frozen_spread_invalid_ou": float(
                (t["fwd_frozen_ou_status"] != "ok").mean())}


# --------------------------------------------------------------------------- metadata
def environment() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "yaml", "pykalman",
                "yfinance"):
        try:
            info[mod] = getattr(importlib.import_module(mod), "__version__", "unknown")
        except ImportError:
            info[mod] = None
    info["backtester"] = backtester_version()
    from . import __version__
    info["pairs"] = __version__
    return info


def write_metadata(ctx: Context, results: dict[str, Any], path: Path) -> Path:
    meta = {"generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "data_mode": ctx.data_mode, "mode": ctx.mode, "dataset": ctx.dataset,
            "seed": ctx.exp["seed"], "environment": environment(),
            "configs": {"experiments": ctx.exp, "screening": ctx.screen_raw,
                        "strategy": ctx.strat, "cohorts": ctx.cohort_cfg},
            "data_provenance": ctx.data_meta,
            "research_data_range": [str(ctx.ohlcv["date"].min().date()),
                                    str(ctx.ohlcv["date"].max().date())],
            "holdout": {"start": ctx.split["holdout_start"], "evaluated_by_this_run": False},
            "timings_seconds": ctx.timings, "results": results}
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"{ctx.data_mode}_{ctx.mode}"
    old = existing.get(key, {})
    if old:  # keep results/timings of steps not re-run this time
        meta["results"] = {**old.get("results", {}), **meta["results"]}
        meta["timings_seconds"] = {**old.get("timings_seconds", {}), **meta["timings_seconds"]}
    existing[key] = meta
    path.write_text(json.dumps(existing, indent=2, default=_jsonable), encoding="utf-8")
    return path


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return str(o)


def run_all(root: Path, data_mode: str = "synthetic", mode: str = "fast",
            steps: tuple[str, ...] | None = None) -> dict[str, Any]:
    t_all = time.perf_counter()
    ctx = build_context(root, data_mode, mode)
    for d in (ctx.fig, ctx.tab, ctx.logs):
        d.mkdir(parents=True, exist_ok=True)
    steps = steps or ("examples", "calibration", "screening", "single_pair", "portfolio",
                      "hedge", "costs", "thresholds", "stability")
    if data_mode == "market":
        steps = tuple(s for s in steps if s not in ("examples", "calibration"))
    out: dict[str, Any] = {}
    fns = {"examples": synthetic_examples, "calibration": lambda c: calibration(c).to_dict("records"),
           "screening": screening, "single_pair": single_pair, "portfolio": portfolio,
           "hedge": lambda c: hedge_comparison(c).to_dict("records"),
           "costs": lambda c: cost_sensitivity(c).to_dict("records"),
           "thresholds": threshold_sensitivity, "stability": stability}
    for s in steps:
        log(f"[{data_mode}/{mode}] {s} ...")
        t0 = time.perf_counter()
        out[s] = fns[s](ctx)
        ctx.timings[s] = time.perf_counter() - t0
        log(f"[{data_mode}/{mode}] {s} done in {ctx.timings[s]:.1f}s")
    ctx.timings["total_last_invocation"] = time.perf_counter() - t_all
    write_metadata(ctx, out, root / "results" / "run_metadata.json")
    what_did_not_work(root)
    return out


# --------------------------------------------------------------------------- hold-out
class HoldoutNotConfirmed(RuntimeError):
    """Raised when the locked hold-out is requested without explicit confirmation."""


def holdout_evaluation(root: Path, data_mode: str = "synthetic", mode: str = "fast",
                       confirm: bool = False) -> dict[str, Any]:
    """Evaluate the FROZEN a-priori configuration once on the locked hold-out.

    Requires ``confirm=True``.  Every evaluation is appended to
    results/logs/holdout_evaluations.jsonl with a running count.  Not called by run_all.
    """
    if not confirm:
        raise HoldoutNotConfirmed("hold-out evaluation requires --confirm (it is logged and "
                                  "cannot be 'un-seen')")
    ctx = build_context(root, data_mode, mode)
    hs = pd.Timestamp(ctx.split["holdout_start"])
    d = root / "data" / ("synthetic" if data_mode == "synthetic" else "market")
    full = D.load_dataset(d, ctx.dataset)
    ctx.ohlcv = full["ohlcv"]
    ctx.__dict__.pop("_close", None)
    ctx.__dict__.pop("_md", None)
    ctx.split["holdout_trading"] = [str(hs.date()), str(ctx.ohlcv["date"].max().date())]
    plans = ctx.plans("holdout")
    res, strat = run(ctx, "holdout", "static_ols", "net", plans=plans, label="HOLD-OUT")
    out = summary(ctx, res, strat, "net")
    logp = ctx.logs / "holdout_evaluations.jsonl"
    n_prev = (sum(1 for line in logp.read_text(encoding="utf-8").splitlines() if line.strip())
              if logp.exists() else 0)
    rec = {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "dataset": ctx.dataset,
           "evaluation_number": n_prev + 1, **{k: out[k] for k in ("sharpe", "cagr",
                                                                    "max_drawdown")}}
    with logp.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=_jsonable) + "\n")
    return out


def what_did_not_work(root: Path) -> pd.DataFrame:
    """Compile negative / inconclusive findings from recorded results (no hand-typed numbers)."""
    meta = json.loads((root / "results" / "run_metadata.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    def add(mode: str, finding: str, evidence: str) -> None:
        rows.append({"data_mode": mode, "finding": finding, "evidence": evidence})

    s = meta.get("synthetic_fast", {}).get("results", {})
    if "screening" in s and "truth" in s["screening"]:
        t = s["screening"]["truth"]
        add("synthetic", "BH at q=0.10 did not control the realised false-discovery proportion",
            f"mean FDP BH {t['fdr_reject_fdp']:.2f}, raw {t['raw_reject_fdp']:.2f}, "
            f"BY {t['by_reject_fdp']:.2f} (BY power {t['by_reject_power']:.1%} vs BH "
            f"{t['fdr_reject_power']:.1%}); EG p-values are anti-conservative under "
            "near-cointegrated nulls (see calibration)")
    cal = root / "results" / "tables" / "syn_00_calibration.csv"
    if cal.exists():
        c = pd.read_csv(cal)
        nc = c[c["experiment"].str.startswith("EG under near-cointegrated")]
        if len(nc):
            add("synthetic", "EG size distortion when the null is 'nearly cointegrated'",
                "; ".join(f"{r.experiment.split(': ')[1]}: {r.rate:.1%}" for r in nc.itertuples()))
        pw = c[c["experiment"].str.startswith("EG 5% power")]
        add("synthetic", "Low power of 12-month EG tests for slow mean reversion",
            "; ".join(f"{r.experiment.split(', ')[1]}: {r.rate:.0%}" for r in pw.itertuples()))
    for mode, key in (("synthetic", "synthetic_fast"), ("market", "market_fast")):
        r = meta.get(key, {}).get("results", {})
        if "hedge" in r:
            h = pd.DataFrame(r["hedge"])
            n = h[h["scenario"] == "net"].set_index("method")
            add(mode, "Kalman filter did not outperform static OLS and traded rarely",
                f"net Sharpe static {n.at['static_ols', 'sharpe']:.2f} / rolling "
                f"{n.at['rolling_ols', 'sharpe']:.2f} / Kalman {n.at['kalman', 'sharpe']:.2f}; "
                f"Kalman trades {int(n.at['kalman', 'completed_pair_trades'])}, share of "
                f"pair-days with |innovation z|>=entry {n.at['kalman', 'share_abs_z_ge_entry']:.1%}")
        if "thresholds" in r:
            t = r["thresholds"]
            add(mode, "Validation-selected thresholds vs out-of-sample",
                f"chosen {t['chosen_on_validation']} with validation Sharpe "
                f"{t['validation_best_sharpe']:.2f}; single OOS evaluation Sharpe "
                f"{t['oos_validation_selected']['sharpe']:.2f}; lookback multiples were inert "
                f"(all hit the 20-session floor): {t['lookback_validation']}")
        if "stability" in r and "share_forward_eg_reject_5pct" in r["stability"]:
            st = r["stability"]
            add(mode, "Selected pairs often did not stay cointegrated",
                f"re-selected next cohort {st['share_reselected_next_cohort']:.0%} (overlapping); "
                f"EG p<0.05 on next non-overlapping 12m {st['share_forward_eg_reject_5pct']:.0%}; "
                f"median half-life {st['median_half_life_form']:.1f} -> "
                f"{st['median_half_life_fwd']:.1f} sessions")
        if "portfolio" in r:
            p = r["portfolio"]
            add(mode, "Headline OOS portfolio (static OLS, a-priori thresholds)",
                f"net excess Sharpe {p['net']['sharpe']:.2f}, gross Sharpe "
                f"{p['gross']['sharpe']:.2f}, net CAGR {p['net']['cagr']:.2%} (includes 2% cash "
                f"interest), avg gross exposure {p['net']['avg_gross_exposure']:.2f}, "
                f"stop-loss exits {p['net']['stop_loss_exits']}")
        if "screening" in r and mode == "market":
            sc = r["screening"]
            add(mode, "Most market formation dates selected no pair",
                f"{sc['cohorts_with_no_selection']} of {sc['n_cohorts']} formation dates; raw EG "
                f"rejections {sc['funnel_totals']['raw_reject']}/{sc['funnel_totals']['candidates']}"
                f" = {sc['funnel_totals']['raw_reject'] / sc['funnel_totals']['candidates']:.1%}"
                " (5% expected under the null)")
    out = pd.DataFrame(rows)
    out.to_csv(root / "results" / "tables" / "20_what_did_not_work.csv", index=False)
    return out
