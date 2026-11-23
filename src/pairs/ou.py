"""Ornstein-Uhlenbeck fitting via AR(1), Hurst exponent and the Lo-MacKinlay variance ratio.

OU / AR(1) mapping (time unit = one trading session unless ``dt`` says otherwise)
-------------------------------------------------------------------------------
The OU process ds = theta (mu - s) dt + sigma dW sampled every dt is exactly

    s_t = a + b s_{t-1} + eps_t,   b = exp(-theta dt),  a = mu (1 - b),
    Var(eps) = sigma^2 (1 - b^2) / (2 theta)

so, for 0 < b < 1:

    theta          = -ln(b) / dt                    (per unit of dt)
    mu             = a / (1 - b)
    half_life      = ln(2) / theta                  (same unit as dt)
    stationary_sd  = sd(eps) / sqrt(1 - b^2)        (sd of the spread LEVEL)
    sigma (diffusion) = sd(eps) * sqrt(2 theta / (1 - b^2))   (per sqrt(unit time))

``stationary_sd`` and the diffusion ``sigma`` are different quantities: the first is the
long-run dispersion of the spread, the second the instantaneous volatility.

b <= 0 is a stationary AR(1) (oscillating for b < 0) but NOT a standard scalar OU
sample: no real theta satisfies exp(-theta dt) = b.  b >= 1 is a unit root/explosive
process with no finite half-life.  Both are flagged invalid; no half-life is returned.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class OUFit:
    status: str  # ok | insufficient_data | constant | negative_b | unit_root_or_explosive
    n: int
    a: float = math.nan
    b: float = math.nan
    b_se: float = math.nan
    resid_sd: float = math.nan
    theta: float = math.nan
    mu: float = math.nan
    half_life: float = math.nan
    half_life_ci: tuple[float, float] = (math.nan, math.nan)
    stationary_sd: float = math.nan
    sigma_diffusion: float = math.nan
    dt: float = 1.0

    @property
    def valid(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["half_life_ci_low"], d["half_life_ci_high"] = d.pop("half_life_ci")
        d["valid"] = self.valid
        return d


def fit_ou(spread: pd.Series | np.ndarray, dt: float = 1.0, min_obs: int = 30,
           ci_level: float = 0.95) -> OUFit:
    """Fit s_t = a + b s_{t-1} + eps by OLS and map to OU parameters.

    Lag alignment: the regression uses pairs (s_{t-1}, s_t) for t = 1..n-1 of the
    observed, NaN-free sequence (NaNs are dropped first; dropping creates irregular
    spacing, so callers should pass a gap-free spread).

    The half-life confidence interval maps the OLS interval for b through the monotone
    transform ln2 * dt / -ln(b); if the interval for b reaches 1 the upper bound is inf.
    This is a large-sample approximation; OLS b is biased downward in finite samples.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    s = np.asarray(pd.Series(spread).dropna(), dtype=float)
    n = len(s)
    if n < min_obs:
        return OUFit("insufficient_data", n, dt=dt)
    if np.nanstd(s) < 1e-12:
        return OUFit("constant", n, dt=dt)
    x, y = s[:-1], s[1:]
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return OUFit("constant", n, dt=dt)
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    a = float(ym - b * xm)
    resid = y - (a + b * x)
    dof = len(y) - 2
    resid_sd = float(np.sqrt((resid ** 2).sum() / dof))
    b_se = resid_sd / math.sqrt(sxx)
    base = {"n": n, "a": a, "b": b, "b_se": b_se, "resid_sd": resid_sd, "dt": dt}
    if b <= 0:
        return OUFit("negative_b", **base)
    if b >= 1:
        return OUFit("unit_root_or_explosive", **base)
    theta = -math.log(b) / dt
    mu = a / (1 - b)
    hl = math.log(2) / theta
    tq = float(stats.t.ppf(0.5 + ci_level / 2, dof))
    lo_b, hi_b = b - tq * b_se, b + tq * b_se

    def hl_of(bb: float) -> float:
        if bb >= 1:
            return math.inf
        if bb <= 0:
            return 0.0
        return math.log(2) * dt / -math.log(bb)

    return OUFit("ok", theta=theta, mu=mu, half_life=hl,
                 half_life_ci=(hl_of(lo_b), hl_of(hi_b)),
                 stationary_sd=resid_sd / math.sqrt(1 - b * b),
                 sigma_diffusion=resid_sd * math.sqrt(2 * theta / (1 - b * b)), **base)


def hurst_exponent(series: pd.Series | np.ndarray, min_lag: int = 2, max_lag: int = 20
                   ) -> dict[str, float]:
    """Exploratory Hurst estimate from the scaling of lagged-difference variance.

    Var(s_{t+tau} - s_t) ~ tau^{2H}  =>  H = slope / 2 of log Var vs log tau over
    tau in [min_lag, max_lag].  For a STATIONARY series the variance saturates at
    2 Var(s), so H falls toward 0 as max_lag grows: the number depends on the lag
    range and is biased in short samples.  H < 0.5 is NOT proof of tradable mean
    reversion; it is a descriptive diagnostic only.
    """
    s = np.asarray(pd.Series(series).dropna(), dtype=float)
    if max_lag <= min_lag or min_lag < 1:
        raise ValueError("need 1 <= min_lag < max_lag")
    if len(s) < 3 * max_lag:
        return {"hurst": math.nan, "r2": math.nan, "min_lag": min_lag, "max_lag": max_lag}
    lags = np.arange(min_lag, max_lag + 1)
    v = np.array([np.var(s[lag:] - s[:-lag]) for lag in lags])
    if np.any(v <= 0):
        return {"hurst": math.nan, "r2": math.nan, "min_lag": min_lag, "max_lag": max_lag}
    lx, ly = np.log(lags), np.log(v)
    slope, _icpt, r, _p, _se = stats.linregress(lx, ly)
    return {"hurst": float(slope / 2.0), "r2": float(r * r), "min_lag": min_lag,
            "max_lag": max_lag}


def variance_ratio_test(levels: pd.Series | np.ndarray, q: int = 5) -> dict[str, float]:
    """Lo-MacKinlay (1988) variance-ratio test on a LEVEL series (e.g. a spread).

    H0: increments are serially uncorrelated (random-walk null; the robust statistic
    z_het allows conditional heteroskedasticity).  VR(q) < 1 indicates negative
    autocorrelation of increments (mean reversion); VR > 1 positive autocorrelation.
    Uses overlapping q-period increments with the bias-corrected variance estimators:

        mu      = (X_T - X_0) / T
        s2_a    = sum (dX_t - mu)^2 / (T - 1)
        m       = q (T - q + 1) (1 - q / T)
        s2_c    = sum_{t=q..T} (X_t - X_{t-q} - q mu)^2 / m
        M       = s2_c / s2_a - 1
        z_hom   = M / sqrt(2 (2q - 1)(q - 1) / (3 q T))
        delta_j = T sum_{t=j+1..T} e_t^2 e_{t-j}^2 / (sum e_t^2)^2,  e_t = dX_t - mu
        theta   = sum_{j=1..q-1} (2 (q - j) / q)^2 delta_j
        z_het   = sqrt(T) M / sqrt(theta)      (delta_j is O(1), so sqrt(T) scales M)

    p-values are two-sided normal approximations (asymptotic).  Calibration of the
    rejection rate under the null is checked by simulation in the test-suite.
    """
    X = np.asarray(pd.Series(levels).dropna(), dtype=float)
    if q < 2:
        raise ValueError("q must be >= 2")
    T = len(X) - 1
    if T < 4 * q:
        return {"vr": math.nan, "z_hom": math.nan, "z_het": math.nan,
                "p_hom": math.nan, "p_het": math.nan, "q": q, "n": T}
    d = np.diff(X)
    mu = (X[-1] - X[0]) / T
    e = d - mu
    s2_a = float((e ** 2).sum() / (T - 1))
    if s2_a <= 0:
        return {"vr": math.nan, "z_hom": math.nan, "z_het": math.nan,
                "p_hom": math.nan, "p_het": math.nan, "q": q, "n": T}
    m = q * (T - q + 1) * (1 - q / T)
    dq = X[q:] - X[:-q] - q * mu
    s2_c = float((dq ** 2).sum() / m)
    vr = s2_c / s2_a
    M = vr - 1.0
    z_hom = M / math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    e2 = e ** 2
    denom = float(e2.sum()) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = T * float((e2[j:] * e2[:-j]).sum()) / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta
    z_het = math.sqrt(T) * M / math.sqrt(theta) if theta > 0 else math.nan
    p = lambda z: float(2 * stats.norm.sf(abs(z))) if math.isfinite(z) else math.nan  # noqa: E731
    return {"vr": vr, "z_hom": z_hom, "z_het": z_het, "p_hom": p(z_hom), "p_het": p(z_het),
            "q": q, "n": T}


def zero_crossings(spread: pd.Series | np.ndarray) -> int:
    """Number of sign changes of the demeaned spread (descriptive)."""
    s = np.asarray(pd.Series(spread).dropna(), dtype=float)
    if len(s) < 2:
        return 0
    sg = np.sign(s - s.mean())
    sg = sg[sg != 0]
    return int((sg[1:] != sg[:-1]).sum())
