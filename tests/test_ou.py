from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pairs import data as D
from pairs.ou import fit_ou, hurst_exponent, variance_ratio_test, zero_crossings


def test_ou_parameter_mapping_exact_on_constructed_ar1() -> None:
    # Build s_t = a + b s_{t-1} + e with e orthogonal to the regressor in-sample is hard;
    # instead check the algebra on the fitted (a, b, sd) values the function reports.
    rng = np.random.default_rng(0)
    b, mu = 0.9, 0.5
    s = np.empty(5000)
    s[0] = mu
    for t in range(1, len(s)):
        s[t] = mu * (1 - b) + b * s[t - 1] + rng.normal(0, 0.01)
    f = fit_ou(s, dt=1.0)
    assert f.valid
    assert f.theta == pytest.approx(-math.log(f.b))
    assert f.mu == pytest.approx(f.a / (1 - f.b))
    assert f.half_life == pytest.approx(math.log(2) / f.theta)
    assert f.stationary_sd == pytest.approx(f.resid_sd / math.sqrt(1 - f.b ** 2))
    assert f.sigma_diffusion == pytest.approx(f.resid_sd * math.sqrt(2 * f.theta / (1 - f.b ** 2)))
    assert abs(f.b - b) < 0.02 and abs(f.mu - mu) < 0.02
    lo, hi = f.half_life_ci
    assert lo < f.half_life < hi


def test_half_life_units_scale_with_dt() -> None:
    u = D.simulate_ar1(3000, 10.0, 0.05, np.random.default_rng(1))
    daily = fit_ou(u, dt=1.0)
    yearly = fit_ou(u, dt=1 / 252)
    assert yearly.half_life == pytest.approx(daily.half_life / 252)
    assert yearly.theta == pytest.approx(daily.theta * 252)
    assert 7 < daily.half_life < 14


def test_invalid_b_never_returns_half_life() -> None:
    rng = np.random.default_rng(2)
    ex = np.empty(400)
    ex[0] = 1.0
    for t in range(1, 400):
        ex[t] = 1.01 * ex[t - 1] + rng.normal(0, 0.1)  # explosive AR(1)
    f = fit_ou(ex)
    assert f.status == "unit_root_or_explosive" and math.isnan(f.half_life) and not f.valid
    alt = np.empty(400)
    alt[0] = 1.0
    for t in range(1, 400):
        alt[t] = -0.6 * alt[t - 1] + rng.normal(0, 0.1)
    g = fit_ou(alt)
    assert g.status == "negative_b" and math.isnan(g.half_life)
    assert fit_ou(np.ones(100)).status == "constant"
    assert fit_ou(np.arange(10.0)).status == "insufficient_data"


def test_half_life_recovery_single_sample_reasonable() -> None:
    u = D.simulate_ar1(2000, 10.0, 0.05, np.random.default_rng(42))
    f = fit_ou(pd.Series(u))
    # single sample: generous tolerance; bias/dispersion is reported by calibration runs
    assert 7.0 < f.half_life < 14.0


def test_hurst_and_variance_ratio_directions() -> None:
    rng = np.random.default_rng(3)
    rw = np.cumsum(rng.normal(size=3000))
    mr = D.simulate_ar1(3000, 5.0, 1.0, rng)
    h_rw, h_mr = hurst_exponent(rw)["hurst"], hurst_exponent(mr)["hurst"]
    assert 0.4 < h_rw < 0.6
    assert h_mr < 0.35
    vr_mr = variance_ratio_test(mr, q=5)
    assert vr_mr["vr"] < 1 and vr_mr["p_het"] < 0.01
    vr_rw = variance_ratio_test(rw, q=5)
    assert abs(vr_rw["z_het"]) < 4  # one draw; calibration tests the rejection rate
    assert math.isnan(hurst_exponent(rw[:30])["hurst"])
    with pytest.raises(ValueError):
        variance_ratio_test(rw, q=1)


def test_variance_ratio_heteroskedastic_and_homoskedastic_agree_under_homoskedasticity() -> None:
    mr = D.simulate_ar1(4000, 8.0, 1.0, np.random.default_rng(4))
    v = variance_ratio_test(mr, q=4)
    assert v["z_het"] == pytest.approx(v["z_hom"], rel=0.25)


def test_zero_crossings() -> None:
    assert zero_crossings(np.array([1, -1, 1, -1, 1.0])) == 4
    assert zero_crossings(np.array([1.0])) == 0


@pytest.mark.slow
def test_variance_ratio_null_rejection_rate() -> None:
    rng = np.random.default_rng(10)
    n = 300
    k = sum(variance_ratio_test(np.cumsum(rng.normal(size=600)), 5)["p_het"] < 0.05
            for _ in range(n))
    # binomial 99% band around 5% for n=300: roughly [2%, 9%]
    assert 0.02 <= k / n <= 0.09
