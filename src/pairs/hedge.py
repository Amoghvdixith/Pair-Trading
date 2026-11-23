"""Hedge-ratio estimation: static OLS, causal rolling OLS and a NumPy Kalman filter.

Naming: ``y`` and ``x`` are OBSERVED (log) prices of the dependent and explanatory legs.
The Kalman LATENT state is ``state = [alpha_t, beta_t]`` - never called x.

Timing conventions
------------------
* Static OLS: coefficients estimated on the formation window and frozen.
* Rolling OLS (``include_current=False``, default): the coefficients used at date t are
  estimated on observations t-W .. t-1, so the residual at t is out-of-sample with
  respect to its own coefficients; z_t = resid_t / sd(in-window residuals).  With
  ``include_current=True`` the window is t-W+1 .. t (the current completed close is part
  of the fit, which shrinks the residual toward zero).
* Kalman: at each t the PREDICTED state (built from data up to t-1) produces the
  innovation e_t = y_t - H_t state_{t|t-1} and its variance F_t = H_t P_{t|t-1} H_t' + R
  BEFORE the update.  e_t / sqrt(F_t) is the innovation z-score.  The posterior residual
  y_t - H_t state_{t|t} uses y_t itself and is mechanically small; it is reported
  separately and must not be used as an out-of-sample z-score.
* Smoothing (RTS) and full-sample EM use future data and are NOT used for trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _align(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    df = pd.concat({"y": pd.Series(y, dtype=float), "x": pd.Series(x, dtype=float)},
                   axis=1).dropna()
    return df


@dataclass(frozen=True)
class HedgeFit:
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    r2: float
    resid_sd: float
    n: int
    resid: pd.Series = field(repr=False)

    def spread(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """Out-of-sample spread with these frozen coefficients."""
        return y - self.alpha - self.beta * x


def ols_hedge(y: pd.Series, x: pd.Series, min_obs: int = 20) -> HedgeFit:
    """OLS of y on [1, x] (joint NaN removal).  Raises on degenerate input."""
    df = _align(y, x)
    n = len(df)
    if n < min_obs:
        raise ValueError(f"need at least {min_obs} joint observations, got {n}")
    xv, yv = df["x"].to_numpy(), df["y"].to_numpy()
    xm, ym = xv.mean(), yv.mean()
    sxx = float(((xv - xm) ** 2).sum())
    if sxx < 1e-14:
        raise ValueError("explanatory series is constant")
    beta = float(((xv - xm) * (yv - ym)).sum() / sxx)
    alpha = float(ym - beta * xm)
    resid = yv - alpha - beta * xv
    sst = float(((yv - ym) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else math.nan
    s2 = float((resid ** 2).sum()) / (n - 2)
    beta_se = math.sqrt(s2 / sxx)
    alpha_se = math.sqrt(s2 * (1.0 / n + xm * xm / sxx))
    return HedgeFit(alpha, beta, alpha_se, beta_se, r2, math.sqrt(s2), n,
                    pd.Series(resid, index=df.index, name="resid"))


def rolling_ols_hedge(y: pd.Series, x: pd.Series, window: int = 60,
                      include_current: bool = False) -> pd.DataFrame:
    """Causal rolling OLS.  Returns alpha, beta, resid_sd, spread and z per date.

    Requires gap-free aligned inputs (NaNs propagate to NaN outputs for affected windows).
    """
    if window < 10:
        raise ValueError("window must be >= 10")
    df = pd.concat({"y": pd.Series(y, dtype=float), "x": pd.Series(x, dtype=float)}, axis=1)
    rx = df["x"].rolling(window, min_periods=window)
    ry = df["y"].rolling(window, min_periods=window)
    mx, my = rx.mean(), ry.mean()
    vx = rx.var(ddof=1)
    cxy = rx.cov(df["y"], ddof=1)
    vy = ry.var(ddof=1)
    beta = cxy / vx.where(vx > 1e-14)
    alpha = my - beta * mx
    # residual variance of the in-window regression (dof = window - 2)
    rss = (vy - beta * cxy) * (window - 1)
    resid_sd = np.sqrt((rss / (window - 2)).clip(lower=0))
    out = pd.DataFrame({"alpha": alpha, "beta": beta, "resid_sd": resid_sd})
    if not include_current:
        out = out.shift(1)
    out["spread"] = df["y"] - out["alpha"] - out["beta"] * df["x"]
    out["z"] = out["spread"] / out["resid_sd"].where(out["resid_sd"] > 1e-12)
    return out


# --------------------------------------------------------------------------- Kalman
@dataclass
class KalmanStep:
    alpha_pred: float
    beta_pred: float
    innovation: float
    innovation_var: float
    z_innovation: float
    alpha: float
    beta: float
    post_resid: float
    loglik: float


class KalmanHedge:
    """Random-walk state [alpha_t, beta_t]; observation y_t = alpha_t + beta_t x_t + v_t.

    state_t = state_{t-1} + w_t,  w_t ~ N(0, Q);   v_t ~ N(0, R).
    Prior: state_{-1} ~ N(m0, P0); the first step predicts P0 + Q before observing y_0.
    Covariance update uses the Joseph form  P = (I-KH) P_pred (I-KH)' + K R K'
    followed by explicit symmetrisation, which keeps P symmetric positive semidefinite
    in floating point.
    """

    def __init__(self, Q: np.ndarray, R: float, m0: np.ndarray, P0: np.ndarray) -> None:
        Q = np.asarray(Q, dtype=float)
        P0 = np.asarray(P0, dtype=float)
        if Q.shape != (2, 2) or P0.shape != (2, 2):
            raise ValueError("Q and P0 must be 2x2")
        if R <= 0:
            raise ValueError("R must be positive")
        for name, M in (("Q", Q), ("P0", P0)):
            if not np.allclose(M, M.T) or np.min(np.linalg.eigvalsh(M)) < -1e-15:
                raise ValueError(f"{name} must be symmetric positive semidefinite")
        self.Q, self.R = Q, float(R)
        self.m = np.asarray(m0, dtype=float).copy()
        self.P = P0.copy()

    def step(self, y: float, x: float) -> KalmanStep:
        if not (math.isfinite(y) and math.isfinite(x)):
            raise ValueError("Kalman step needs finite y and x (skip missing bars upstream)")
        H = np.array([1.0, x])
        m_pred = self.m  # random-walk transition
        P_pred = self.P + self.Q
        e = float(y - H @ m_pred)
        F = float(H @ P_pred @ H + self.R)
        K = P_pred @ H / F
        self.m = m_pred + K * e
        IKH = np.eye(2) - np.outer(K, H)
        P = IKH @ P_pred @ IKH.T + self.R * np.outer(K, K)
        self.P = 0.5 * (P + P.T)
        ll = -0.5 * (math.log(2 * math.pi * F) + e * e / F)
        return KalmanStep(float(m_pred[0]), float(m_pred[1]), e, F, e / math.sqrt(F),
                          float(self.m[0]), float(self.m[1]),
                          float(y - H @ self.m), ll)


def delta_to_Q(delta: float) -> np.ndarray:
    """Chan (2013) parameterisation Q = delta / (1 - delta) * I (equal scale for both states)."""
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    return delta / (1.0 - delta) * np.eye(2)


def kalman_filter(y: pd.Series, x: pd.Series, Q: np.ndarray, R: float, m0: np.ndarray,
                  P0: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Run the filter forward (causal) over jointly observed dates.

    Returns a frame indexed by date (alpha_pred, beta_pred, innovation, innovation_var,
    z_innovation, alpha, beta, post_resid) and the total log-likelihood.
    """
    df = _align(y, x)
    kf = KalmanHedge(Q, R, m0, P0)
    recs = []
    ll = 0.0
    for t, (yy, xx) in zip(df.index, df[["y", "x"]].to_numpy(), strict=True):
        st = kf.step(float(yy), float(xx))
        ll += st.loglik
        recs.append(st.__dict__ | {"date": t})
    out = pd.DataFrame(recs).set_index("date").drop(columns="loglik")
    return out, ll


@dataclass(frozen=True)
class KalmanTuning:
    delta: float
    R: float
    m0: tuple[float, float]
    P0_diag: tuple[float, float]
    loglik: float
    grid: dict[str, float]

    @property
    def Q(self) -> np.ndarray:
        return delta_to_Q(self.delta)


def tune_kalman(y: pd.Series, x: pd.Series, delta_grid: list[float],
                prior_var: tuple[float, float] = (1.0, 1.0)) -> KalmanTuning:
    """Choose delta by maximum innovation log-likelihood on the GIVEN (formation) data.

    R is fixed at the formation OLS residual variance; the prior mean is the formation OLS
    estimate.  The caller must pass formation data only.
    """
    fit = ols_hedge(y, x)
    R = max(fit.resid_sd ** 2, 1e-10)
    m0 = np.array([fit.alpha, fit.beta])
    P0 = np.diag(prior_var)
    grid: dict[str, float] = {}
    best: tuple[float, float] | None = None
    for d in delta_grid:
        _, ll = kalman_filter(y, x, delta_to_Q(d), R, m0, P0)
        grid[f"{d:g}"] = ll
        if best is None or ll > best[1]:
            best = (d, ll)
    assert best is not None
    return KalmanTuning(best[0], R, (fit.alpha, fit.beta), prior_var, best[1], grid)


def pykalman_reference(y: pd.Series, x: pd.Series, Q: np.ndarray, R: float, m0: np.ndarray,
                       P0: np.ndarray) -> pd.DataFrame | None:
    """Filtered state means from pykalman with IDENTICAL model, or None if not installed.

    pykalman observes y_0 against state_0 ~ N(initial_mean, initial_cov) without a prior
    transition; our filter predicts P0 + Q first.  We therefore pass initial_cov = P0 + Q
    so both describe the same prior for the state at t = 0.
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        return None
    df = _align(y, x)
    obs_mat = np.stack([np.ones(len(df)), df["x"].to_numpy()], axis=1)[:, None, :]
    kf = KalmanFilter(transition_matrices=np.eye(2), observation_matrices=obs_mat,
                      transition_covariance=Q, observation_covariance=np.array([[R]]),
                      initial_state_mean=np.asarray(m0, dtype=float),
                      initial_state_covariance=np.asarray(P0, dtype=float) + Q)
    means, _covs = kf.filter(df["y"].to_numpy()[:, None])
    return pd.DataFrame(means, index=df.index, columns=["alpha", "beta"])


def describe_kalman(t: KalmanTuning) -> dict[str, Any]:
    return {"delta": t.delta, "Q_diag": float(delta_to_Q(t.delta)[0, 0]), "R": t.R,
            "m0": list(t.m0), "P0_diag": list(t.P0_diag), "formation_loglik": t.loglik,
            "loglik_grid": t.grid}
