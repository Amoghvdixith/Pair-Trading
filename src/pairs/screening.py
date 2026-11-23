"""Cointegration screening: unit-root diagnostics, Engle-Granger, Johansen and FDR control.

Stages are kept distinct in the audit table:

1. ``raw_reject``      - EG p-value < 0.05 (single-test decision, NOT multiplicity-adjusted)
2. ``fdr_reject``      - rejected after BH (or BY) across the formation-date family
3. ``passes_filters``  - FDR-rejected AND economic / mean-reversion filters pass
4. ``selected``        - final portfolio selection (ranking + one-pair-per-stock)

Engle-Granger p-values come from ``statsmodels.tsa.stattools.coint``, which uses
MacKinnon (1994, 2010) response-surface distributions for residual-based tests with an
ESTIMATED cointegrating vector (two variables, constant).  Ordinary ADF critical values
are too lenient for estimated residuals and are never used here.

Johansen trace statistics are compared to tabulated critical values only (statsmodels
provides 90/95/99% values, not p-values).  They are never fed into the FDR procedure.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .data import log_prices
from .hedge import ols_hedge
from .ou import fit_ou, hurst_exponent, variance_ratio_test, zero_crossings
from .universe import UniverseRules, candidate_pairs, filter_universe

# --------------------------------------------------------------------------- unit roots


def unit_root_diagnostics(series: pd.Series, autolag: str = "AIC") -> dict[str, Any]:
    """ADF (constant) on levels and first differences.

    Failing to reject a unit root in levels is NOT proof that a series is I(1); the ADF
    test has low power against near-unit-root alternatives.  The label is descriptive.
    """
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 30 or s.std() < 1e-12:
        return {"status": "insufficient_or_constant"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # statsmodels return-type notice
        lv = adfuller(s.to_numpy(), regression="c", autolag=autolag)
        df = adfuller(s.diff().dropna().to_numpy(), regression="c", autolag=autolag)
    label = ("consistent with I(1)" if lv[1] > 0.05 and df[1] < 0.05 else
             "levels look stationary" if lv[1] <= 0.05 else "differences not clearly I(0)")
    return {"status": "ok", "adf_level_stat": float(lv[0]), "adf_level_p": float(lv[1]),
            "adf_diff_stat": float(df[0]), "adf_diff_p": float(df[1]), "label": label}


# --------------------------------------------------------------------------- Engle-Granger
@dataclass(frozen=True)
class EGResult:
    status: str  # ok | insufficient_data | constant | degenerate_perfect_fit | failed
    y: str
    x: str
    n: int
    stat: float = math.nan
    pvalue: float = math.nan
    crit_1: float = math.nan
    crit_5: float = math.nan
    crit_10: float = math.nan
    alpha: float = math.nan
    beta: float = math.nan
    message: str = ""


def engle_granger(y: pd.Series, x: pd.Series, trend: str = "c", autolag: str | None = "aic",
                  maxlag: int | None = None, min_obs: int = 60,
                  collinearity_tol: float = 1e-10, names: tuple[str, str] = ("y", "x")
                  ) -> EGResult:
    """Engle-Granger two-step test of H0: no cointegration (y regressed on x).

    Deterministic term ``trend='c'`` (constant in the cointegrating regression); the ADF
    regression on residuals selects its lag by ``autolag`` (AIC) up to ``maxlag``
    (statsmodels default 12 (n/100)^{1/4}).  p-values: MacKinnon (2010) approximation for
    N = 2 variables, as implemented by statsmodels.
    """
    df = pd.concat({"y": pd.Series(y, dtype=float), "x": pd.Series(x, dtype=float)},
                   axis=1).dropna()
    n = len(df)
    ny, nx = names
    if n < min_obs:
        return EGResult("insufficient_data", ny, nx, n)
    if df["y"].std() < 1e-12 or df["x"].std() < 1e-12:
        return EGResult("constant", ny, nx, n)
    try:
        fit = ols_hedge(df["y"], df["x"])
    except ValueError as exc:
        return EGResult("failed", ny, nx, n, message=str(exc))
    if not math.isfinite(fit.r2) or 1 - fit.r2 < collinearity_tol:
        return EGResult("degenerate_perfect_fit", ny, nx, n, alpha=fit.alpha, beta=fit.beta,
                        message="near-perfect collinearity: residual test undefined")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, crit = coint(df["y"].to_numpy(), df["x"].to_numpy(), trend=trend,
                                  method="aeg", maxlag=maxlag, autolag=autolag)
    except Exception as exc:  # pragma: no cover - statsmodels internal failure
        return EGResult("failed", ny, nx, n, alpha=fit.alpha, beta=fit.beta, message=str(exc))
    return EGResult("ok", ny, nx, n, float(stat), float(p), float(crit[0]), float(crit[1]),
                    float(crit[2]), fit.alpha, fit.beta)


def orient(a: str, b: str, prices: pd.DataFrame, rule: str) -> tuple[str, str]:
    """Return (y, x) - dependent first - by a rule fixed BEFORE testing.

    ``alphabetical``: y = alphabetically first symbol (pure a-priori).
    ``max_variance``: y = leg with larger formation log-price variance (uses levels, not
    any test outcome).
    """
    if rule == "alphabetical":
        return (a, b) if a <= b else (b, a)
    if rule == "max_variance":
        va, vb = prices[a].var(), prices[b].var()
        return (a, b) if va >= vb else (b, a)
    raise ValueError(f"unknown orientation rule {rule!r}")


# --------------------------------------------------------------------------- Johansen
def johansen(y: pd.Series, x: pd.Series, det_order: int = 0, k_ar_diff: int = 1,
             signif: float = 0.05) -> dict[str, Any]:
    """Johansen trace / max-eigenvalue rank diagnostic for a two-series system.

    Rank decision (sequential trace test at the tabulated ``signif`` level):
    rank 0 = no cointegration; rank 1 = one cointegrating relation; rank 2 = full rank,
    i.e. the VAR looks stationary in levels (both series I(0)) - not a classical pair.
    The eigenvector for the largest eigenvalue is normalised so the first (y) weight is 1;
    ``beta_johansen = -v2 / v1``.  It is an estimate, not a validated tradable basket.
    """
    col = {0.10: 0, 0.05: 1, 0.01: 2}
    if signif not in col:
        raise ValueError("signif must be 0.10, 0.05 or 0.01 (tabulated values only)")
    df = pd.concat({"y": pd.Series(y, dtype=float), "x": pd.Series(x, dtype=float)},
                   axis=1).dropna()
    if len(df) < 60 or df.std().min() < 1e-12:
        return {"johansen_status": "insufficient_or_constant"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels casts tiny imaginary parts
            res = coint_johansen(df.to_numpy(), det_order, k_ar_diff)
    except Exception as exc:  # pragma: no cover
        return {"johansen_status": f"failed: {exc}"}
    c = col[signif]
    tr, cvt = np.real(res.lr1), res.cvt
    if tr[0] < cvt[0, c]:
        rank = 0
    elif tr[1] < cvt[1, c]:
        rank = 1
    else:
        rank = 2
    v = np.real(res.evec[:, 0])
    beta_j = float(-v[1] / v[0]) if abs(v[0]) > 1e-12 else math.nan
    return {"johansen_status": "ok", "trace_r0": float(tr[0]), "trace_r1": float(tr[1]),
            "trace_cv_r0": float(cvt[0, c]), "trace_cv_r1": float(cvt[1, c]),
            "maxeig_r0": float(np.real(res.lr2[0])), "maxeig_cv_r0": float(res.cvm[0, c]),
            "johansen_rank": rank, "beta_johansen": beta_j,
            "johansen_det_order": det_order, "johansen_k_ar_diff": k_ar_diff}


# --------------------------------------------------------------------------- FDR
def _clean_p(pvalues: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float).copy()
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    p[~np.isfinite(p)] = 1.0  # failed fits count in the family as non-discoveries
    return p


def _step_up(p: np.ndarray, q: float, c_m: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < q < 1:
        raise ValueError("q must lie in (0, 1)")
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m * c_m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= q, adj


def benjamini_hochberg(pvalues: np.ndarray | list[float], q: float = 0.05
                       ) -> tuple[np.ndarray, np.ndarray]:
    """BH step-up: reject H_(1..k) for the largest k with p_(k) <= k q / m.

    Returns (reject mask, BH-adjusted p-values) in the input order.  NaN p-values (failed
    fits) are treated as 1 and stay in the family size m.  BH controls FDR <= q for
    independent or positively regression-dependent (PRDS) test statistics.
    """
    return _step_up(_clean_p(pvalues), q, 1.0)


def benjamini_yekutieli(pvalues: np.ndarray | list[float], q: float = 0.05
                        ) -> tuple[np.ndarray, np.ndarray]:
    """BY step-up: BH with q / c(m), c(m) = sum_{i<=m} 1/i; valid under ANY dependence."""
    p = _clean_p(pvalues)
    cm = float(np.sum(1.0 / np.arange(1, len(p) + 1))) if len(p) else 1.0
    return _step_up(p, q, cm)


# --------------------------------------------------------------------------- screen
@dataclass
class ScreenConfig:
    price_transform: str = "log"
    universe: UniverseRules = field(default_factory=UniverseRules)
    orientation_rule: str = "alphabetical"
    direction_policy: str = "predeclared"
    trend: str = "c"
    autolag: str | None = "aic"
    maxlag: int | None = None
    collinearity_tol: float = 1e-10
    johansen_det_order: int = 0
    johansen_k_ar_diff: int = 1
    johansen_signif: float = 0.05
    fdr_method: str = "bh"
    fdr_q: float = 0.10
    half_life_min: float = 2.0
    half_life_max: float = 60.0
    beta_min: float = 0.2
    beta_max: float = 5.0
    min_zero_crossings: int = 6
    max_pairs: int = 10
    one_pair_per_stock: bool = True
    rank_by: str = "eg_pvalue"
    run_johansen: bool = True

    def __post_init__(self) -> None:
        if self.price_transform not in ("log", "raw"):
            raise ValueError("price_transform must be 'log' or 'raw'")
        if self.direction_policy not in ("predeclared", "both_bonferroni"):
            raise ValueError("direction_policy must be predeclared or both_bonferroni")
        if self.fdr_method not in ("bh", "by"):
            raise ValueError("fdr_method must be bh or by")
        if self.rank_by not in ("eg_pvalue", "half_life"):
            raise ValueError("rank_by must be eg_pvalue or half_life")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScreenConfig:
        eg, jo, fdr = d.get("engle_granger", {}), d.get("johansen", {}), d.get("fdr", {})
        fl, se = d.get("filters", {}), d.get("selection", {})
        return cls(
            price_transform=d.get("price_transform", "log"),
            universe=UniverseRules.from_dict(d.get("universe")),
            orientation_rule=eg.get("orientation_rule", "alphabetical"),
            direction_policy=eg.get("direction_policy", "predeclared"),
            trend=eg.get("trend", "c"), autolag=eg.get("autolag", "aic"),
            maxlag=eg.get("maxlag"), collinearity_tol=float(eg.get("collinearity_tol", 1e-10)),
            johansen_det_order=int(jo.get("det_order", 0)),
            johansen_k_ar_diff=int(jo.get("k_ar_diff", 1)),
            johansen_signif=float(jo.get("signif", 0.05)),
            fdr_method=fdr.get("method", "bh"), fdr_q=float(fdr.get("q", 0.10)),
            half_life_min=float(fl.get("half_life_min", 2.0)),
            half_life_max=float(fl.get("half_life_max", 60.0)),
            beta_min=float(fl.get("beta_min", 0.2)), beta_max=float(fl.get("beta_max", 5.0)),
            min_zero_crossings=int(fl.get("min_zero_crossings", 6)),
            max_pairs=int(se.get("max_pairs", 10)),
            one_pair_per_stock=bool(se.get("one_pair_per_stock", True)),
            rank_by=se.get("rank_by", "eg_pvalue"),
        )


@dataclass
class ScreenResult:
    cutoff: pd.Timestamp
    formation_start: pd.Timestamp
    universe_audit: pd.DataFrame
    candidates: pd.DataFrame
    selected: pd.DataFrame
    funnel: dict[str, int]


def _transform(px: pd.DataFrame, how: str) -> pd.DataFrame:
    return log_prices(px) if how == "log" else px.astype(float)


def screen_pairs(close: pd.DataFrame, volume: pd.DataFrame, meta: pd.DataFrame,
                 cutoff: pd.Timestamp, cfg: ScreenConfig,
                 membership: pd.DataFrame | None = None) -> ScreenResult:
    """Screen same-industry candidates on FORMATION data only.

    ``close`` and ``volume`` must end at or before ``cutoff`` (asserted).  The tested
    family for FDR is every candidate pair formed from eligible symbols at this cutoff.
    """
    cutoff = pd.Timestamp(cutoff)
    if close.index.max() > cutoff or volume.index.max() > cutoff:
        raise ValueError("screen_pairs received data after the formation cutoff")
    elig, uaudit = filter_universe(close, volume, meta, cutoff, cfg.universe, membership)
    cands = candidate_pairs(elig, meta, cfg.universe.candidate_rule)
    lp = _transform(close[elig], cfg.price_transform) if elig else close.iloc[:, :0]
    rets = log_prices(close[elig]).diff() if elig else close.iloc[:, :0]
    rows = []
    for rec in cands.to_dict("records"):
        a, b = rec["a"], rec["b"]
        y, x = orient(a, b, lp, cfg.orientation_rule)
        kw = {"trend": cfg.trend, "autolag": cfg.autolag, "maxlag": cfg.maxlag,
              "collinearity_tol": cfg.collinearity_tol}
        eg_yx = engle_granger(lp[y], lp[x], names=(y, x), **kw)
        eg_xy = engle_granger(lp[x], lp[y], names=(x, y), **kw)
        if cfg.direction_policy == "predeclared":
            p_pair = eg_yx.pvalue
        else:
            ps = [p for p in (eg_yx.pvalue, eg_xy.pvalue) if math.isfinite(p)]
            p_pair = min(1.0, 2 * min(ps)) if ps else math.nan
        row: dict[str, Any] = {
            **rec, "y": y, "x": x, "eg_status": eg_yx.status, "eg_stat": eg_yx.stat,
            "eg_pvalue_yx": eg_yx.pvalue, "eg_pvalue_xy": eg_xy.pvalue,
            "eg_crit_5": eg_yx.crit_5, "p_pair": p_pair, "alpha": eg_yx.alpha,
            "beta": eg_yx.beta, "return_corr": float(rets[y].corr(rets[x])),
        }
        if eg_yx.status == "ok":
            spread = lp[y] - eg_yx.alpha - eg_yx.beta * lp[x]
            spread = spread.dropna()
            ou = fit_ou(spread)
            row.update({f"ou_{k}": v for k, v in ou.as_dict().items()
                        if k in ("status", "b", "half_life", "half_life_ci_low",
                                 "half_life_ci_high", "stationary_sd", "valid")})
            row["spread_sd"] = float(spread.std())
            row["zero_crossings"] = zero_crossings(spread)
            row["hurst"] = hurst_exponent(spread)["hurst"]
            vr = variance_ratio_test(spread, q=5)
            row["vr5"], row["vr5_p_het"] = vr["vr"], vr["p_het"]
        if cfg.run_johansen:
            row.update(johansen(lp[y], lp[x], cfg.johansen_det_order, cfg.johansen_k_ar_diff,
                                cfg.johansen_signif))
        rows.append(row)
    cand = pd.DataFrame(rows)
    funnel = {"symbols_total": int(len(uaudit)), "symbols_eligible": len(elig),
              "candidates": int(len(cand))}
    if cand.empty:
        funnel.update(raw_reject=0, fdr_reject=0, passes_filters=0, selected=0)
        return ScreenResult(cutoff, close.index.min(), uaudit, cand, cand.copy(), funnel)
    fdr = benjamini_hochberg if cfg.fdr_method == "bh" else benjamini_yekutieli
    rej, adj = fdr(cand["p_pair"].to_numpy(), cfg.fdr_q)
    cand["p_adj"] = adj
    cand["raw_reject"] = cand["p_pair"].fillna(1.0) < 0.05
    cand["fdr_reject"] = rej
    hl = cand.get("ou_half_life", pd.Series(np.nan, index=cand.index))
    valid = cand.get("ou_valid", pd.Series(False, index=cand.index)).fillna(False).astype(bool)
    zc = cand.get("zero_crossings", pd.Series(0, index=cand.index)).fillna(0)
    reasons = []
    for i in cand.index:
        r = []
        if not cand.at[i, "fdr_reject"]:
            r.append("not FDR-significant")
        if not valid.at[i]:
            r.append("OU fit invalid")
        elif not cfg.half_life_min <= hl.at[i] <= cfg.half_life_max:
            r.append("half-life out of range")
        bt = cand.at[i, "beta"]
        if not (math.isfinite(bt) and cfg.beta_min <= bt <= cfg.beta_max):
            r.append("beta outside policy range")
        if zc.at[i] < cfg.min_zero_crossings:
            r.append("too few zero crossings")
        reasons.append("; ".join(r))
    cand["filter_reasons"] = reasons
    cand["passes_filters"] = cand["filter_reasons"] == ""
    key = "p_pair" if cfg.rank_by == "eg_pvalue" else "ou_half_life"
    ranked = cand[cand["passes_filters"]].sort_values([key, "pair"], kind="mergesort")
    used: set[str] = set()
    sel_idx: list[Any] = []
    for i, r in ranked.iterrows():
        if len(sel_idx) >= cfg.max_pairs:
            break
        if cfg.one_pair_per_stock and (r["a"] in used or r["b"] in used):
            continue
        sel_idx.append(i)
        used.update((r["a"], r["b"]))
    cand["selected"] = cand.index.isin(sel_idx)
    cand.insert(0, "cutoff", cutoff)
    funnel.update(raw_reject=int(cand["raw_reject"].sum()),
                  fdr_reject=int(cand["fdr_reject"].sum()),
                  passes_filters=int(cand["passes_filters"].sum()),
                  selected=int(cand["selected"].sum()))
    selected = cand[cand["selected"]].reset_index(drop=True)
    return ScreenResult(cutoff, close.index.min(), uaudit, cand, selected, funnel)
