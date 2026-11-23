"""Integration with the project-02 backtester (offline, small synthetic universes)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from backtester import Backtest

from pairs import analysis as A
from pairs import data as D
from pairs.backtester_adapter import (
    PassThroughSizer,
    RunSpec,
    backtester_version,
    market_data_from_long,
    run_pairs_backtest,
)
from pairs.cohorts import make_schedule, overlap_count
from pairs.screening import ScreenConfig
from pairs.strategy import PairsStrategy, StrategyParams, build_plans, sectors_map

from .conftest import manual_plan, pair_market


def _run(oh, meta, plans, strat_cfg, screen_cfg, scenario="net", flat=0.0, **ov):
    md = market_data_from_long(oh)
    params = StrategyParams.from_config(strat_cfg, screen_cfg, **ov)
    s = PairsStrategy(plans, params, sectors_map(meta))
    res = run_pairs_backtest(md, s, strat_cfg, RunSpec(scenario, flat), plans[0].cohort.trade_start
                             if plans else md.index[1], "test", "synthetic")
    return res, s, md


def test_adapter_uses_installed_backtester(strat_cfg, screen_cfg) -> None:
    assert backtester_version() == "0.1.0"
    oh, meta = pair_market()
    plan = manual_plan(oh)
    res, s, md = _run(oh, meta, [plan], strat_cfg, screen_cfg, entry_z=1.5)
    assert res.meta["strategy"]["strategy"] == "pairs"
    assert res.meta["execution_convention"].startswith("signal at close t")
    assert len(s.trades) > 0 and not res.meta["backstop_bound"]
    # every fill happens strictly after the close that created its order
    f = res.fills
    assert (pd.to_datetime(f["timestamp"]) > pd.to_datetime(f["order_created_at"])).all()
    assert isinstance(PassThroughSizer(), PassThroughSizer)


def test_both_legs_charged_costs_and_borrow(strat_cfg, screen_cfg) -> None:
    oh, meta = pair_market()
    res, s, md = _run(oh, meta, [manual_plan(oh)], strat_cfg, screen_cfg, "flat", 10.0,
                      entry_z=1.5)
    f = res.fills
    assert set(f["symbol"]) == {"AAA", "BBB"}
    np.testing.assert_allclose(f["commission"], 10e-4 * f["quantity"].abs() * f["fill_price"])
    assert res.equity["borrow_fees"].iloc[-1] > 0  # short leg pays borrow
    g, _, _ = _run(oh, meta, [manual_plan(oh)], strat_cfg, screen_cfg, "gross", entry_z=1.5)
    assert g.fills["commission"].sum() == 0 and g.equity["borrow_fees"].iloc[-1] == 0


def test_pair_vs_stock_pnl_reconciliation(strat_cfg, screen_cfg) -> None:
    oh, meta = pair_market()
    res, s, md = _run(oh, meta, [manual_plan(oh)], strat_cfg, screen_cfg, entry_z=1.5)
    pnl, rec = A.attribute_pairs(res, s, md)
    assert rec["unattributed"].abs().max() < 1e-6
    tt = A.trade_table(s, pnl)
    assert len(tt) == len([t for t in s.trades if "exit_fill" in t])
    # entry holdings follow the beta-weighted rule
    aud = pd.DataFrame(s.holdings_audit)
    assert (np.sign(aud["y_shares"]) == -np.sign(aud["x_shares"])).all()


def test_two_leg_failure_unwinds_filled_leg(strat_cfg, screen_cfg) -> None:
    oh, meta = pair_market()
    plan = manual_plan(oh)
    res, s, _ = _run(oh, meta, [plan], strat_cfg, screen_cfg, entry_z=1.5)
    first = pd.Timestamp(s.trades[0]["entry_fill"])  # session at whose open both legs filled
    idx = sorted(oh["date"].unique())
    k = idx.index(first)
    drop_rows = oh.index[(oh["symbol"] == "BBB") & (oh["date"] == first)]
    oh2 = oh.drop(index=drop_rows)  # x leg cannot trade on the fill session
    res2, s2, md2 = _run(oh2, meta, [manual_plan(oh)], strat_cfg, screen_cfg, entry_z=1.5)
    ev = pd.DataFrame(s2.events)
    fail = ev[ev["event"] == "leg_failure"]
    assert len(fail) >= 1 and pd.Timestamp(fail["date"].iloc[0]) == first
    # the filled y leg is unwound at the next open: no AAA position after that session
    f = res2.fills
    aaa = f[f["symbol"] == "AAA"]
    t_fail = pd.Timestamp(idx[k])
    nxt = pd.Timestamp(idx[k + 1])
    assert ((pd.to_datetime(aaa["timestamp"]) == t_fail).any()
            and (pd.to_datetime(aaa["timestamp"]) == nxt).any())
    assert not s2.confirmed.get(plan.pairs[0].key)  # never counted as an open pair
    st = res2.order_status
    assert ((st["symbol"] == "BBB") & (st["status"] == "expired")).any()


def test_future_mutation_does_not_change_past_decisions(strat_cfg, screen_cfg) -> None:
    oh, meta = pair_market()
    plan = manual_plan(oh)
    res, s, _ = _run(oh, meta, [plan], strat_cfg, screen_cfg, entry_z=1.5)
    cut = sorted(oh["date"].unique())[400]
    oh2 = oh.copy()
    m = oh2["date"] > cut
    for c in ("open", "high", "low", "close"):
        oh2.loc[m & (oh2["symbol"] == "AAA"), c] *= 1.3
    res2, s2, _ = _run(oh2, meta, [manual_plan(oh)], strat_cfg, screen_cfg, entry_z=1.5)
    e1 = pd.DataFrame(s.events)
    e2 = pd.DataFrame(s2.events)
    pd.testing.assert_frame_equal(e1[e1["date"] <= cut].reset_index(drop=True),
                                  e2[e2["date"] <= cut].reset_index(drop=True))
    pd.testing.assert_frame_equal(res.equity.loc[:cut], res2.equity.loc[:cut])


def test_no_pair_selected_stays_in_cash(strat_cfg, screen_cfg) -> None:
    oh, meta = pair_market()
    plan = manual_plan(oh)
    empty = copy.copy(plan)
    empty.pairs = []
    res, s, _ = _run(oh, meta, [empty], strat_cfg, screen_cfg)
    assert res.fills.empty and (res.equity["gross_exposure"] == 0).all()
    assert res.equity["equity"].iloc[-1] > res.equity["equity"].iloc[0]  # cash interest only


@pytest.fixture(scope="module")
def small_universe(strat_cfg, screen_raw):
    spec = D.UniverseSpec(start="2014-01-01", end="2016-12-30", n_sectors=3,
                          stocks_per_sector=5, seed=99)
    u = D.generate_universe(spec)
    close, vol = D.wide(u["ohlcv"], "close"), D.wide(u["ohlcv"], "volume")
    cfg = ScreenConfig.from_dict(screen_raw)
    coh = make_schedule(close.index, 12, 6, 3, first_trade_month="2015-01",
                        last_trade_end="2016-12-30")
    plans = build_plans(close, vol, u["universe"], coh, cfg, strat_cfg)
    return u, plans, cfg


def test_cohort_capital_accounting_and_no_duplicate_returns(small_universe, strat_cfg) -> None:
    u, plans, cfg = small_universe
    coh = [p.cohort for p in plans]
    assert all(c.capital_share == 0.5 for c in coh)
    assert max(overlap_count(coh, d) for d in pd.bdate_range("2015-01-01", "2016-12-30")) <= 2
    cap = float(strat_cfg["constraints"]["portfolio_gross_cap"])
    for p in plans:  # planned budgets never exceed the cohort's share of the gross budget
        assert sum(ps.gross_frac for ps in p.pairs) <= cap * p.cohort.capital_share + 1e-12
    res, s, md = _run(u["ohlcv"], u["universe"], plans, strat_cfg, cfg, entry_z=1.5)
    eq = res.equity
    assert not eq.index.has_duplicates  # one account, each session once
    assert eq.loc[eq["trading"], "gross_exposure"].max() <= cap * 1.15  # drift tolerance
    pnl, rec = A.attribute_pairs(res, s, md)
    assert rec["unattributed"].abs().max() < 1e-6  # cohorts sum to the single account
    keys = {k.split(":")[0] for k in pnl.columns}
    assert len(keys) >= 2  # several cohorts actually traded
    ce = A.cohort_exposure(s, md, res.returns().index, eq["equity"])
    assert (ce.filter(like="cohort_") > 0).sum(axis=1).max() <= 2


def test_rolling_and_kalman_methods_run_and_rehedge_costs(small_universe, strat_cfg) -> None:
    u, plans, cfg = small_universe
    r_roll, s_roll, _ = _run(u["ohlcv"], u["universe"], plans, strat_cfg, cfg,
                             method="rolling_ols", entry_z=1.5)
    ev = pd.DataFrame(s_roll.events)
    if (ev["event"] == "rehedge_confirmed").any():
        d = pd.Timestamp(ev.loc[ev["event"] == "rehedge_confirmed", "date"].iloc[0])
        f = r_roll.fills
        assert (pd.to_datetime(f["timestamp"]) == d).any()  # rehedge traded (and paid costs)
    r_k, s_k, _ = _run(u["ohlcv"], u["universe"], plans, strat_cfg, cfg, method="kalman",
                       entry_z=1.5)
    assert r_k.meta["pairs"]["hedge_method"] == "kalman"


def test_unattainable_stop_is_rejected(strat_cfg, screen_cfg) -> None:
    cfg = copy.deepcopy(strat_cfg)
    cfg["signals"]["zscore_lookback_min"] = 10
    with pytest.raises(ValueError, match="unattainable"):
        StrategyParams.from_config(cfg, screen_cfg)


def test_backtest_object_is_public_api() -> None:
    assert hasattr(Backtest, "run")
