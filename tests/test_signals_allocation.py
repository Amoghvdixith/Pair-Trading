from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pairs.allocation import (
    LegTargets,
    Limits,
    admit_scale,
    check_book,
    engine_weight_for_shares,
    inverse_vol_gross,
    notional_split,
    pair_holdings,
    raw_price_holdings,
)
from pairs.signals import (
    Action,
    PairState,
    PairStateMachine,
    SignalConfig,
    max_attainable_z,
    rolling_zscore,
    run_state_machine,
    zscore_last,
)


# ------------------------------------------------------------------ z-scores
def test_zscore_include_current_convention() -> None:
    s = pd.Series([1.0, 2, 3, 4, 10])
    z = rolling_zscore(s, 5, include_current=True)
    v = s.to_numpy()
    assert z.iloc[-1] == pytest.approx((10 - v.mean()) / v.std(ddof=1))
    assert zscore_last(v) == pytest.approx(z.iloc[-1])
    z2 = rolling_zscore(s, 4, include_current=False)
    w = v[:4]
    assert z2.iloc[-1] == pytest.approx((10 - w.mean()) / w.std(ddof=1))


def test_zscore_zero_volatility_is_nan_not_inf() -> None:
    z = rolling_zscore(pd.Series([1.0] * 10), 5)
    assert z.iloc[-1] != z.iloc[-1]
    assert math.isnan(zscore_last(np.ones(5)))


def test_zscore_bound_with_current_included() -> None:
    for L in (5, 10, 20):
        v = np.zeros(L)
        v[-1] = 1e6  # most extreme possible last point
        assert zscore_last(v) == pytest.approx(max_attainable_z(L))
    assert max_attainable_z(10) < 3.0 < max_attainable_z(20)


# ------------------------------------------------------------------ state machine
CFG = SignalConfig(entry_z=2.0, exit_z=0.0, stop_z=3.5, time_stop=5, cooldown=2)


def test_hand_built_z_path_gives_expected_sequence() -> None:
    z = pd.Series([0.0, 1.0, 2.1, 1.5, 0.5, -0.1, 0.0, 0.0, -2.2, -1.0, 0.1])
    out = run_state_machine(z, CFG)
    acts = out["action"].tolist()
    assert acts == ["none", "none", "open_short", "none", "none", "close", "none", "none",
                    "open_long", "none", "close"]
    assert out["reason"].iloc[5] == "take_profit"


def test_entry_requires_confirmed_fills() -> None:
    sm = PairStateMachine(CFG)
    assert sm.on_close(2.5).action is Action.OPEN_SHORT
    assert sm.state is PairState.ENTRY_PENDING and not sm.is_open
    assert sm.on_close(0.0).action is Action.NONE  # pending: nothing happens
    assert sm.confirm(False) == "leg_failure"
    assert sm.state is PairState.FLAT and sm.cooldown_left == CFG.cooldown
    assert sm.on_close(2.5).reason == "cooldown"


def test_stop_precedence_and_suspension() -> None:
    sm = PairStateMachine(CFG)
    sm.on_close(-2.5)
    sm.confirm(True)
    assert sm.state is PairState.LONG_SPREAD
    # stop beats take-profit logic and time stop; then suspension
    d = sm.on_close(-3.6)
    assert d.action is Action.CLOSE and d.reason == "stop_loss"
    sm.confirm(True)
    assert sm.state is PairState.SUSPENDED
    assert sm.on_close(-2.5).action is Action.NONE
    # expiry beats breakdown beats stop
    sm2 = PairStateMachine(CFG)
    sm2.on_close(2.5)
    sm2.confirm(True)
    assert sm2.on_close(4.0, breakdown=True, expiring=True).reason == "window_expiry"
    sm3 = PairStateMachine(CFG)
    sm3.on_close(2.5)
    sm3.confirm(True)
    assert sm3.on_close(4.0, breakdown=True).reason == "breakdown"
    sm3.confirm(True)
    assert sm3.state is PairState.SUSPENDED


def test_time_stop_missing_data_and_stop_range_blocks_entry() -> None:
    sm = PairStateMachine(CFG)
    sm.on_close(2.4)
    sm.confirm(True)
    reasons = [sm.on_close(1.5).reason for _ in range(5)]
    assert reasons[-1] == "time_stop"
    sm4 = PairStateMachine(SignalConfig(time_stop=50, max_missing=2))
    sm4.on_close(2.4)
    sm4.confirm(True)
    r = [sm4.on_close(math.nan, missing=True).reason for _ in range(3)]
    assert r == ["missing_hold", "missing_hold", "missing_data"]
    flat = PairStateMachine(CFG)
    assert flat.on_close(3.6).reason == "stop_range_blocks_entry"
    assert flat.on_close(math.nan).action is Action.NONE


def test_cross_mode_requires_crossing() -> None:
    sm = PairStateMachine(SignalConfig(entry_mode="cross"))
    assert sm.on_close(2.5, 2.2).action is Action.NONE  # already beyond: no cross
    assert sm.on_close(2.5, 1.9).action is Action.OPEN_SHORT


def test_signal_config_validation() -> None:
    with pytest.raises(ValueError):
        SignalConfig(entry_z=2.0, exit_z=2.5)
    with pytest.raises(ValueError):
        SignalConfig(entry_z=4.0, stop_z=3.5)


# ------------------------------------------------------------------ holdings
def test_log_beta_to_holdings() -> None:
    lt = pair_holdings(beta=1.5, gross=100_000, direction=1, price_y=50.0, price_x=20.0)
    # N_y = 100k/2.5 = 40k -> 800 sh; N_x = -1.5*40k = -60k -> -3000 sh
    assert (lt.y_shares, lt.x_shares) == (800, -3000)
    assert lt.y_notional == pytest.approx(40_000) and lt.x_notional == pytest.approx(-60_000)
    # NOT dollar neutral unless beta = 1: |net|/gross = |1-b|/(1+b) = 0.2
    assert lt.dollar_imbalance == pytest.approx(0.2)
    # share ratio is beta * P_y / P_x, not beta
    assert abs(lt.x_shares / lt.y_shares) == pytest.approx(1.5 * 50 / 20)
    dn = pair_holdings(1.5, 100_000, -1, 50.0, 20.0, rule="dollar_neutral")
    assert dn.dollar_imbalance == pytest.approx(0.0) and dn.y_shares < 0 < dn.x_shares
    with pytest.raises(ValueError):
        notional_split(-0.5, 1.0, 1)
    assert raw_price_holdings(1.5, 100) == (100, -150)


def test_rounding_intended_vs_realised() -> None:
    lt = pair_holdings(beta=1.0, gross=1_000, direction=1, price_y=333.0, price_x=7.0)
    assert lt.y_shares == 1 and lt.x_shares == -71
    assert lt.rounding_gap < 0  # rounding toward zero never increases gross
    assert isinstance(lt, LegTargets)


def test_limits_and_admission() -> None:
    sectors = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}
    lim = Limits(portfolio_gross_cap=1.0, net_cap=0.2, name_cap=0.15, sector_gross_cap=0.5)
    cur = {"A": 0.10, "B": -0.10}
    new = {"A": 0.10, "B": -0.10}
    k, why = admit_scale(cur, new, sectors, lim)
    assert 0.25 <= k < 0.6 and why  # name cap 0.15 binds: 0.10 + k*0.10 <= 0.15
    assert not check_book({s: cur.get(s, 0) + k * new.get(s, 0) for s in "AB"}, sectors, lim)
    k2, _ = admit_scale({"A": 0.15, "B": -0.15}, new, sectors, lim)
    assert k2 == 0.0
    viol = check_book({"C": 0.3, "D": 0.3}, sectors, lim)
    assert any("net" in v for v in viol) and any("name" in v for v in viol)
    g = inverse_vol_gross({"p1": 0.01, "p2": 0.02}, budget=0.3, cap=0.15)
    assert g["p1"] == pytest.approx(0.15) and g["p2"] == pytest.approx(0.1)


def test_engine_weight_reproduces_exact_shares() -> None:
    rng = np.random.default_rng(0)
    for _ in range(2000):
        q = int(rng.integers(-5000, 5000))
        p = float(rng.uniform(1, 900))
        eq = float(rng.uniform(1e5, 1e7))
        w = engine_weight_for_shares(q, p, eq)
        assert math.trunc(w * eq / p) == q
