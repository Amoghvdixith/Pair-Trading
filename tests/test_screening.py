from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.adfvalues import mackinnoncrit

from pairs import data as D
from pairs.cohorts import make_schedule
from pairs.hedge import ols_hedge
from pairs.screening import (
    benjamini_hochberg,
    benjamini_yekutieli,
    engle_granger,
    johansen,
    orient,
    screen_pairs,
)
from pairs.strategy import build_cohort_plan


def test_cointegrating_coefficient_recovery() -> None:
    df, t = D.cointegrated_pair(2000, seed=21)
    fit = ols_hedge(df["y"], df["x"])
    # OLS is super-consistent here; tolerance ~ several SEs of a single sample
    assert abs(fit.beta - t.beta) < 0.1
    eg = engle_granger(df["y"], df["x"])
    assert eg.status == "ok" and eg.pvalue < 0.01


def test_eg_uses_cointegration_not_adf_critical_values() -> None:
    df, _ = D.cointegrated_pair(500, seed=2)
    eg = engle_granger(df["y"], df["x"])
    # MacKinnon (2010) residual-based 5% value for N=2 with constant ~ -3.34,
    # clearly more negative than the plain ADF value (~ -2.87)
    assert -3.45 < eg.crit_5 < -3.30
    adf_5 = mackinnoncrit(N=1, regression="c", nobs=500)[1]
    assert eg.crit_5 < adf_5 - 0.3


def test_orientation_rule_is_predeclared() -> None:
    px = pd.DataFrame({"B": [1.0, 2, 3, 4], "A": [1.0, 1.1, 1.2, 1.3]})
    assert orient("B", "A", px, "alphabetical") == ("A", "B")
    assert orient("A", "B", px, "max_variance") == ("B", "A")
    with pytest.raises(ValueError):
        orient("A", "B", px, "best_pvalue")


def test_eg_direction_asymmetry_retained_in_audit(screen_cfg) -> None:
    df, _ = D.cointegrated_pair(300, seed=9)
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.DataFrame({"AAA": np.exp(df["y"].to_numpy()), "BBB": np.exp(df["x"].to_numpy())},
                         index=idx)
    vol = pd.DataFrame(1e7, index=idx, columns=close.columns)
    meta = pd.DataFrame({"symbol": ["AAA", "BBB"], "sector": ["S", "S"],
                         "industry": ["S", "S"], "currency": ["USD", "USD"]})
    res = screen_pairs(close, vol, meta, idx[-1], screen_cfg)
    row = res.candidates.iloc[0]
    assert row["y"] == "AAA" and np.isfinite(row["eg_pvalue_xy"])
    assert row["p_pair"] == row["eg_pvalue_yx"]  # predeclared: one p-value, no min()
    from dataclasses import replace
    res2 = screen_pairs(close, vol, meta, idx[-1],
                        replace(screen_cfg, direction_policy="both_bonferroni"))
    r2 = res2.candidates.iloc[0]
    assert r2["p_pair"] == pytest.approx(min(1.0, 2 * min(r2["eg_pvalue_yx"],
                                                          r2["eg_pvalue_xy"])))


def test_bh_hand_example() -> None:
    p = [0.01, 0.04, 0.03, 0.20]
    rej, adj = benjamini_hochberg(p, 0.05)
    # sorted 0.01,0.03,0.04,0.20 vs k q/m = .0125,.025,.0375,.05 -> only k=1 passes
    assert rej.tolist() == [True, False, False, False]
    np.testing.assert_allclose(adj, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20])
    rej, _ = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205], 0.05)
    # classic step-up: largest k with p_(k) <= k*0.05/8 is k=2 (0.008 <= 0.0125) ...
    # but p_(5)=0.042 > 0.03125 and p_(3)=0.039 > 0.01875 -> k=2
    assert rej.sum() == 2


def test_bh_step_up_rejects_below_largest_k() -> None:
    # p_(1)=0.02 > q/m=0.0125 alone, but p_(4)=0.05 <= 4q/4 -> all four rejected
    rej, _ = benjamini_hochberg([0.02, 0.03, 0.04, 0.05], 0.05)
    assert rej.all()


def test_by_is_more_conservative_and_nan_counts_in_family() -> None:
    p = [0.001, 0.01, 0.02, 0.03, np.nan]
    rb, ab = benjamini_hochberg(p, 0.10)
    ry, ay = benjamini_yekutieli(p, 0.10)
    assert ry.sum() <= rb.sum()
    assert np.all(ay >= ab - 1e-15)
    assert not rb[-1] and ab[-1] == 1.0
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.2], 0.1)


def test_johansen_rank_handling() -> None:
    df, _ = D.cointegrated_pair(1500, seed=5)
    j = johansen(df["y"], df["x"])
    assert j["johansen_rank"] >= 1
    assert abs(j["beta_johansen"] - 1.3) < 0.15
    rng = np.random.default_rng(0)
    st = pd.DataFrame({"y": rng.normal(size=800), "x": rng.normal(size=800)})
    assert johansen(st["y"], st["x"])["johansen_rank"] == 2  # both I(0): full rank
    with pytest.raises(ValueError):
        johansen(df["y"], df["x"], signif=0.02)


def test_degenerate_inputs() -> None:
    idx = pd.RangeIndex(200)
    x = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=200)), index=idx)
    assert engle_granger(pd.Series(1.0, index=idx), x).status == "constant"
    assert engle_granger(2 * x + 1, x).status == "degenerate_perfect_fit"
    assert engle_granger(x.iloc[:30], x.iloc[:30] + 1).status == "insufficient_data"
    y = x + np.random.default_rng(2).normal(size=200)
    y.iloc[::7] = np.nan  # NaNs are dropped jointly, not filled
    assert engle_granger(y, x).n == int(y.notna().sum())
    assert johansen(pd.Series(1.0, index=idx), x)["johansen_status"] != "ok"


def _universe():
    spec = D.UniverseSpec(start="2014-01-01", end="2016-06-30", n_sectors=2,
                          stocks_per_sector=4, seed=17)
    u = D.generate_universe(spec)
    return D.wide(u["ohlcv"], "close"), D.wide(u["ohlcv"], "volume"), u["universe"]


def test_formation_window_boundaries_and_future_invariance(screen_cfg, strat_cfg) -> None:
    close, vol, meta = _universe()
    coh = make_schedule(close.index, 12, 6, 3, first_trade_month="2015-01")[0]
    assert coh.cutoff < coh.trade_start
    assert coh.formation_start == close.index[close.index >= "2014-01-01"][0]
    with pytest.raises(ValueError):
        screen_pairs(close, vol, meta, coh.cutoff, screen_cfg)  # unsliced data refused
    p1 = build_cohort_plan(close, vol, meta, coh, screen_cfg, strat_cfg)
    mutated = close.copy()
    after = mutated.index > coh.cutoff
    mutated.loc[after] = mutated.loc[after] * np.exp(
        np.random.default_rng(0).normal(0, 0.3, mutated.loc[after].shape))
    vol2 = vol.copy()
    vol2.loc[after] = 1.0
    p2 = build_cohort_plan(mutated, vol2, meta, coh, screen_cfg, strat_cfg)
    pd.testing.assert_frame_equal(p1.screen.candidates, p2.screen.candidates)
    assert p1.pairs == p2.pairs


def test_no_selected_pairs_is_valid(screen_cfg) -> None:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=260)
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.02, (260, 4)), axis=0)) * 50,
                         index=idx, columns=list("ABCD"))
    vol = pd.DataFrame(1e7, index=idx, columns=close.columns)
    meta = pd.DataFrame({"symbol": list("ABCD"), "sector": ["S"] * 4, "industry": ["S"] * 4,
                         "currency": ["USD"] * 4})
    from dataclasses import replace
    res = screen_pairs(close, vol, meta, idx[-1], replace(screen_cfg, fdr_q=0.01))
    assert res.funnel["candidates"] == 6
    assert res.selected.empty or res.funnel["selected"] <= 2


@pytest.mark.slow
def test_eg_null_rejection_rate_independent_random_walks() -> None:
    """Calibration (not a per-sample decision test): H0 = no cointegration is TRUE."""
    n = 300
    k = sum(engle_granger(*(lambda d: (d["y"], d["x"]))(
        D.independent_random_walks(252, seed=1000 + i)[0])).pvalue < 0.05 for i in range(n))
    # 99% binomial band around 5% for n=300 is roughly [2.1%, 8.9%]
    assert 0.02 <= k / n <= 0.09
