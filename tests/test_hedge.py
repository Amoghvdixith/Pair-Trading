from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pairs import data as D
from pairs.hedge import (
    KalmanHedge,
    delta_to_Q,
    kalman_filter,
    ols_hedge,
    pykalman_reference,
    rolling_ols_hedge,
    tune_kalman,
)


def test_kalman_single_step_hand_calculation() -> None:
    Q = np.diag([0.1, 0.2])
    R = 0.5
    m0 = np.array([1.0, 2.0])
    P0 = np.eye(2)
    kf = KalmanHedge(Q, R, m0, P0)
    y, x = 10.0, 3.0
    st = kf.step(y, x)
    H = np.array([1.0, x])
    Pp = P0 + Q
    e = y - H @ m0  # 10 - (1 + 6) = 3
    F = H @ Pp @ H + R  # 1.1 + 9*1.2 + 0.5 = 12.4
    K = Pp @ H / F
    assert st.innovation == pytest.approx(3.0)
    assert st.innovation_var == pytest.approx(12.4)
    np.testing.assert_allclose([st.alpha, st.beta], m0 + K * e)
    P_std = Pp - np.outer(K, H) @ Pp
    np.testing.assert_allclose(kf.P, P_std, atol=1e-12)  # Joseph == standard form
    assert st.z_innovation == pytest.approx(3.0 / math.sqrt(12.4))
    assert st.alpha_pred == 1.0 and st.beta_pred == 2.0  # prediction BEFORE the update


def test_innovation_uses_predicted_state_only() -> None:
    df, _ = D.cointegrated_pair(300, seed=1)
    out, _ = kalman_filter(df["y"], df["x"], delta_to_Q(1e-5), 0.002, np.array([0.5, 1.3]),
                           np.eye(2))
    # innovation at t equals y_t - (alpha_pred_t + beta_pred_t x_t), and the predicted
    # state at t equals the posterior at t-1 (random walk)
    e = df["y"] - out["alpha_pred"] - out["beta_pred"] * df["x"]
    np.testing.assert_allclose(out["innovation"], e, atol=1e-12)
    np.testing.assert_allclose(out["alpha_pred"].iloc[1:], out["alpha"].iloc[:-1])
    # changing y at t+1 cannot change anything at or before t
    y2 = df["y"].copy()
    y2.iloc[200] += 5.0
    out2, _ = kalman_filter(y2, df["x"], delta_to_Q(1e-5), 0.002, np.array([0.5, 1.3]),
                            np.eye(2))
    pd.testing.assert_frame_equal(out.iloc[:200], out2.iloc[:200])
    assert (out["post_resid"].abs().mean() < out["innovation"].abs().mean())


def test_kalman_covariance_symmetric_psd_over_long_run() -> None:
    df, _ = D.time_varying_beta_pair(3000, seed=2)
    kf = KalmanHedge(delta_to_Q(1e-4), 1e-4, np.zeros(2), np.eye(2) * 1e3)
    for yy, xx in df[["y", "x"]].to_numpy():
        kf.step(float(yy), float(xx))
        assert np.allclose(kf.P, kf.P.T, atol=0)
        assert np.linalg.eigvalsh(kf.P).min() >= -1e-14
    with pytest.raises(ValueError):
        KalmanHedge(np.array([[1, 2], [0, 1.0]]), 1.0, np.zeros(2), np.eye(2))
    with pytest.raises(ValueError):
        KalmanHedge(np.eye(2), 0.0, np.zeros(2), np.eye(2))


def test_pykalman_equivalence_identical_model() -> None:
    pytest.importorskip("pykalman")
    df, _ = D.time_varying_beta_pair(800, seed=3)
    Q, R, m0, P0 = delta_to_Q(1e-5), 1e-3, np.array([0.0, 1.0]), np.eye(2)
    ours, _ = kalman_filter(df["y"], df["x"], Q, R, m0, P0)
    ref = pykalman_reference(df["y"], df["x"], Q, R, m0, P0)
    assert ref is not None
    np.testing.assert_allclose(ours[["alpha", "beta"]].to_numpy(), ref.to_numpy(), atol=1e-8)


def test_kalman_tuning_uses_given_window_only() -> None:
    df, _ = D.time_varying_beta_pair(1000, seed=4)
    t1 = tune_kalman(df["y"].iloc[:400], df["x"].iloc[:400], [1e-6, 1e-5, 1e-4])
    y2 = df["y"].copy()
    y2.iloc[400:] += 3.0
    t2 = tune_kalman(y2.iloc[:400], df["x"].iloc[:400], [1e-6, 1e-5, 1e-4])
    assert t1 == t2


def test_rolling_ols_is_causal_and_excludes_current() -> None:
    df, _ = D.cointegrated_pair(400, seed=5)
    r = rolling_ols_hedge(df["y"], df["x"], 60, include_current=False)
    t = 250
    w = df.iloc[t - 60:t]  # observations t-60 .. t-1
    fit = ols_hedge(w["y"], w["x"])
    assert r["beta"].iloc[t] == pytest.approx(fit.beta)
    assert r["resid_sd"].iloc[t] == pytest.approx(fit.resid_sd)
    assert r["spread"].iloc[t] == pytest.approx(df["y"].iloc[t] - fit.alpha
                                                - fit.beta * df["x"].iloc[t])
    y2 = df["y"].copy()
    y2.iloc[t:] += 10
    r2 = rolling_ols_hedge(y2, df["x"], 60)
    assert r2["beta"].iloc[t] == pytest.approx(r["beta"].iloc[t])  # future change irrelevant
    inc = rolling_ols_hedge(df["y"], df["x"], 60, include_current=True)
    assert inc["beta"].iloc[t - 1] == pytest.approx(r["beta"].iloc[t])


def test_ols_rejects_degenerate() -> None:
    with pytest.raises(ValueError):
        ols_hedge(pd.Series(np.arange(50.0)), pd.Series(np.ones(50)))
    with pytest.raises(ValueError):
        ols_hedge(pd.Series(np.arange(5.0)), pd.Series(np.arange(5.0)))
