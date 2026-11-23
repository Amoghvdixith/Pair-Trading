"""Synthetic data generators, OHLCV construction and dataset I/O.

All synthetic data are labelled ``data_mode = "synthetic"``.  They demonstrate that the
software behaves as designed; synthetic trading returns are NOT evidence of market alpha.

Log-price conventions
---------------------
Generators work in natural-log prices.  Prices are ``exp(log price)`` with levels chosen
in a numerically sensible range (roughly 10-500).

OHLCV construction (assumed intraday dynamics)
----------------------------------------------
Only the close path comes from the statistical model.  The open of session t is drawn
from a Brownian bridge between the closes of t-1 and t: with ``f`` the overnight share
of daily variance,

    log O_t ~ N( log C_{t-1} + f (log C_t - log C_{t-1}),  f (1 - f) sigma_t^2 )

which is the conditional law of a latent continuous log-price path at time fraction f.
High/low extend max/min(open, close) by half-normal excursions.  This is a coherent
generating model, not leakage: a strategy only ever sees the open of t before the close
of t (the backtester's point-in-time view enforces this).  Volume is drawn from a
lognormal around a target dollar volume and is independent of future prices.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------- primitives
def ar1_coefficient(half_life: float, dt: float = 1.0) -> float:
    """AR(1) coefficient b with half-life ``half_life`` (same time units as ``dt``)."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return float(0.5 ** (dt / half_life))


def simulate_ar1(n: int, half_life: float, stationary_sd: float,
                 rng: np.random.Generator, start_stationary: bool = True) -> np.ndarray:
    """Zero-mean AR(1)/sampled-OU path with the given half-life and stationary sd."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if stationary_sd < 0:
        raise ValueError("stationary_sd must be >= 0")
    b = ar1_coefficient(half_life)
    eps_sd = stationary_sd * np.sqrt(1.0 - b * b)
    out = np.empty(n)
    out[0] = rng.normal(0.0, stationary_sd) if start_stationary else 0.0
    eps = rng.normal(0.0, eps_sd, n)
    for t in range(1, n):
        out[t] = b * out[t - 1] + eps[t]
    return out


def random_walk(n: int, sd: float, rng: np.random.Generator, start: float = 0.0,
                drift: float = 0.0) -> np.ndarray:
    steps = rng.normal(drift, sd, n)
    steps[0] = 0.0
    return start + np.cumsum(steps)


# --------------------------------------------------------------------------- examples A-E
@dataclass(frozen=True)
class PairTruth:
    """Known data-generating parameters of a synthetic pair (log prices)."""

    kind: str
    alpha: float | None = None
    beta: float | None = None
    half_life: float | None = None
    noise_sd: float | None = None
    break_at: int | None = None
    notes: str = ""


def cointegrated_pair(n: int = 2000, alpha: float = 0.5, beta: float = 1.3,
                      half_life: float = 10.0, noise_sd: float = 0.05,
                      x_start: float = np.log(40.0), x_vol: float = 0.015,
                      seed: int = 0) -> tuple[pd.DataFrame, PairTruth]:
    """Example A: x random walk; y = alpha + beta*x + AR(1) noise (log prices)."""
    rng = np.random.default_rng(seed)
    x = random_walk(n, x_vol, rng, start=x_start)
    u = simulate_ar1(n, half_life, noise_sd, rng)
    y = alpha + beta * x + u
    df = pd.DataFrame({"x": x, "y": y, "u": u}, index=_bdays(n))
    return df, PairTruth("cointegrated", alpha, beta, half_life, noise_sd)


def independent_random_walks(n: int = 2000, vol: float = 0.015, seed: int = 0
                             ) -> tuple[pd.DataFrame, PairTruth]:
    """Example B: two independent log random walks (no cointegration)."""
    rng = np.random.default_rng(seed)
    x = random_walk(n, vol, rng, start=np.log(40.0))
    y = random_walk(n, vol, rng, start=np.log(60.0))
    return pd.DataFrame({"x": x, "y": y}, index=_bdays(n)), PairTruth("independent")


def correlated_random_walks(n: int = 2000, rho: float = 0.8, vol: float = 0.015,
                            seed: int = 0) -> tuple[pd.DataFrame, PairTruth]:
    """Example C: random walks with correlated innovations, |rho| < 1 (NOT cointegrated).

    The innovation covariance ``vol^2 [[1, rho], [rho, 1]]`` is nonsingular, so no linear
    combination of the levels is stationary even though returns are highly correlated.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly inside (-1, 1) for a nonsingular covariance")
    rng = np.random.default_rng(seed)
    cov = vol ** 2 * np.array([[1.0, rho], [rho, 1.0]])
    steps = rng.multivariate_normal(np.zeros(2), cov, size=n)
    steps[0] = 0.0
    lv = np.cumsum(steps, axis=0) + np.log([40.0, 60.0])
    df = pd.DataFrame({"x": lv[:, 0], "y": lv[:, 1]}, index=_bdays(n))
    return df, PairTruth("correlated_not_cointegrated", notes=f"rho={rho}")


def breaking_pair(n: int = 2000, break_at: int = 1200, alpha: float = 0.5, beta: float = 1.3,
                  half_life: float = 10.0, noise_sd: float = 0.05, break_vol: float = 0.012,
                  seed: int = 0) -> tuple[pd.DataFrame, PairTruth]:
    """Example D: cointegrated until ``break_at``; afterwards y accumulates a random walk."""
    if not 0 < break_at < n:
        raise ValueError("break_at must be inside the sample")
    df, _ = cointegrated_pair(n, alpha, beta, half_life, noise_sd, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    drift = np.zeros(n)
    drift[break_at:] = np.cumsum(rng.normal(0.0, break_vol, n - break_at))
    df["y"] = df["y"] + drift
    df["u"] = df["u"] + drift
    return df, PairTruth("breakdown", alpha, beta, half_life, noise_sd, break_at)


def time_varying_beta_pair(n: int = 2000, beta_start: float = 1.0, beta_end: float = 1.6,
                           alpha: float = 0.2, half_life: float = 8.0, noise_sd: float = 0.03,
                           seed: int = 0) -> tuple[pd.DataFrame, PairTruth]:
    """Example E: y = alpha + beta_t x + AR(1) noise with a slowly drifting beta_t.

    Note: with a time-varying beta the "spread" y - beta_t x is stationary only around the
    TRUE path beta_t; an estimated path can absorb divergence (see docs/hedge_to_holdings).
    """
    rng = np.random.default_rng(seed)
    x = random_walk(n, 0.012, rng, start=np.log(30.0))
    s = np.linspace(0, 1, n)
    beta_t = beta_start + (beta_end - beta_start) * (3 * s ** 2 - 2 * s ** 3)  # smoothstep
    u = simulate_ar1(n, half_life, noise_sd, rng)
    y = alpha + beta_t * x + u
    df = pd.DataFrame({"x": x, "y": y, "beta_true": beta_t, "u": u}, index=_bdays(n))
    return df, PairTruth("time_varying_beta", alpha, None, half_life, noise_sd,
                         notes=f"beta {beta_start}->{beta_end} (smoothstep)")


def _bdays(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# --------------------------------------------------------------------------- OHLCV
def ohlcv_from_close(log_close: np.ndarray, rng: np.random.Generator,
                     dollar_volume: float = 2.0e7, overnight_share: float = 0.3,
                     range_scale: float = 0.6) -> dict[str, np.ndarray]:
    """Coherent OHLCV bars from a log-close path (see module docstring)."""
    lc = np.asarray(log_close, dtype=float)
    n = len(lc)
    r = np.diff(lc, prepend=lc[0])
    sig = np.full(n, np.std(r[1:]) if n > 2 else 0.01)
    f = overnight_share
    lo = np.empty(n)
    lo[0] = lc[0] + rng.normal(0.0, 0.002)
    prev = lc[:-1]
    mean = prev + f * (lc[1:] - prev)
    lo[1:] = mean + rng.normal(0.0, 1.0, n - 1) * np.sqrt(f * (1 - f)) * sig[1:]
    o, c = np.exp(lo), np.exp(lc)
    up = np.abs(rng.normal(0.0, range_scale * sig))
    dn = np.abs(rng.normal(0.0, range_scale * sig))
    h = np.maximum(o, c) * np.exp(up)
    low = np.minimum(o, c) * np.exp(-dn)
    vol = dollar_volume / c * np.exp(rng.normal(0.0, 0.35, n))
    return {"open": o, "high": h, "low": low, "close": c, "volume": np.round(vol)}


# --------------------------------------------------------------------------- sector universe
@dataclass
class UniverseSpec:
    """Synthetic sector universe with a market factor and known cointegration structure.

    log P_i,t = c_i + lambda_i * F_k(i),t + u_i,t + w_i,t   with
      F_k = M + S_k            (market random walk + sector random walk)
      u_i = AR(1) noise         (half-life 3-20 sessions)
      w_i = idiosyncratic random walk: zero for "linked" stocks, switched on after a
            break date for "breaking" stocks, always on for "unlinked" stocks.

    Two linked stocks i, j of the same sector are cointegrated with log-price
    beta = lambda_i / lambda_j (until either breaks).  Unlinked stocks are correlated
    with their sector but NOT cointegrated with anything.  The last sector is a pure-null
    sector (all unlinked) to exercise the false-discovery machinery.
    """

    start: str = "2012-01-02"
    end: str = "2023-12-29"
    n_sectors: int = 4
    stocks_per_sector: int = 6
    seed: int = 20261122
    market_vol: float = 0.010
    market_drift: float = 0.0003
    sector_vol: float = 0.008
    noise_sd_range: tuple[float, float] = (0.02, 0.035)
    half_life_range: tuple[float, float] = (3.0, 20.0)
    idio_vol: float = 0.009
    overnight_share: float = 0.3
    sector_names: list[str] = field(default_factory=lambda: [
        "Energy", "Banks", "Miners", "Retail", "Utilities", "Telecom", "Rail", "Insurers"])


def generate_universe(spec: UniverseSpec) -> dict[str, Any]:
    """Generate the synthetic sector universe.

    Returns a dict with ``ohlcv`` (long frame), ``universe`` (metadata), ``benchmark``
    (market index close series), ``truth`` (per-stock generating parameters) and
    ``pair_truth`` (which same-sector pairs are cointegrated over which dates).
    """
    if spec.n_sectors < 2 or spec.stocks_per_sector < 4:
        raise ValueError("need >= 2 sectors and >= 4 stocks per sector")
    if spec.n_sectors > len(spec.sector_names):
        raise ValueError("not enough sector names")
    rng = np.random.default_rng(spec.seed)
    idx = pd.bdate_range(spec.start, spec.end)
    n = len(idx)
    mkt = random_walk(n, spec.market_vol, rng, drift=spec.market_drift)
    rows: list[pd.DataFrame] = []
    meta, truth = [], []
    for k in range(spec.n_sectors):
        sector = spec.sector_names[k]
        code = sector[0].upper() + sector[1].upper()
        null_sector = k == spec.n_sectors - 1
        s_k = random_walk(n, spec.sector_vol, rng)
        f_k = mkt + s_k
        for m in range(spec.stocks_per_sector):
            sym = f"{code}{m + 1:02d}"
            lam = float(rng.uniform(0.7, 1.5))
            hl = float(rng.uniform(*spec.half_life_range))
            nsd = float(rng.uniform(*spec.noise_sd_range))
            u = simulate_ar1(n, hl, nsd, rng)
            w = np.zeros(n)
            # roles: 0-2 linked; 3 breaks; others unlinked; null sector all unlinked
            if null_sector or m >= 4:
                role, brk = "unlinked", None
                w = random_walk(n, spec.idio_vol, rng)
            elif m == 3:
                role = "breaking"
                brk = int(rng.integers(int(0.35 * n), int(0.75 * n)))
                w[brk:] = np.cumsum(rng.normal(0.0, spec.idio_vol, n - brk))
            else:
                role, brk = "linked", None
            c = float(np.log(rng.uniform(15, 250)))
            lc = c + lam * f_k + u + w
            dv = float(np.exp(rng.uniform(np.log(1.5e7), np.log(2e8))))
            if m == spec.stocks_per_sector - 1 and k == 1:
                dv = 1.0e6  # deliberately illiquid name (fails the liquidity filter)
            bars = ohlcv_from_close(lc, rng, dollar_volume=dv,
                                    overnight_share=spec.overnight_share)
            df = pd.DataFrame(bars, index=idx)
            if m == spec.stocks_per_sector - 2 and k == 0:
                # a stretch of missing data (exchange suspension) for the missing-data rule
                a = int(0.20 * n)
                df.iloc[a:a + 60] = np.nan
            df.insert(0, "symbol", sym)
            df.index.name = "date"
            rows.append(df.reset_index())
            meta.append({"symbol": sym, "sector": sector, "industry": sector,
                         "currency": "USD", "price_unit": "USD"})
            truth.append({"symbol": sym, "sector": sector, "role": role, "lambda": lam,
                          "half_life": hl, "noise_sd": nsd,
                          "break_date": str(idx[brk].date()) if brk is not None else None})
    ohlcv = pd.concat(rows, ignore_index=True)
    ohlcv = ohlcv.dropna(subset=["close"]).reset_index(drop=True)
    bench = pd.Series(100.0 * np.exp(mkt), index=idx, name="MKT")
    tr = pd.DataFrame(truth)
    return {"ohlcv": ohlcv, "universe": pd.DataFrame(meta), "benchmark": bench,
            "truth": tr, "pair_truth": pair_truth_table(tr)}


def pair_truth_table(truth: pd.DataFrame) -> pd.DataFrame:
    """Same-sector pairs with their true cointegration status (log prices)."""
    out = []
    for _sector, g in truth.groupby("sector", sort=True):
        recs = g.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                linked = {a["role"], b["role"]} <= {"linked", "breaking"}
                brk = [r["break_date"] for r in (a, b) if isinstance(r["break_date"], str)]
                out.append({
                    "pair": f"{a['symbol']}-{b['symbol']}", "a": a["symbol"], "b": b["symbol"],
                    "sector": a["sector"], "cointegrated": linked,
                    "cointegrated_until": min(brk) if (linked and brk) else None,
                    "true_beta_a_on_b": a["lambda"] / b["lambda"] if linked else None,
                })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- I/O
def write_synthetic_dataset(spec: UniverseSpec, directory: str | Path, name: str
                            ) -> dict[str, str]:
    """Generate and write ``<name>.csv`` (long OHLCV) plus metadata files."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out = generate_universe(spec)
    out["ohlcv"].to_csv(d / f"{name}.csv", index=False, float_format="%.6f")
    out["universe"].to_csv(d / f"{name}_universe.csv", index=False)
    out["benchmark"].to_frame().to_csv(d / f"{name}_benchmark.csv", float_format="%.6f")
    out["truth"].to_csv(d / f"{name}_truth.csv", index=False)
    out["pair_truth"].to_csv(d / f"{name}_pair_truth.csv", index=False)
    meta = {
        "data_mode": "synthetic",
        "generator": "pairs.data.generate_universe",
        "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()},
        "price_adjustment": "none (synthetic); prices are exp(log-price model)",
        "currency": "USD (synthetic label)",
        "ohlc_model": "Brownian-bridge open between consecutive closes; see pairs.data",
        "classification_retrieved": "synthetic labels, fixed at generation",
        "point_in_time_membership": "all symbols listed for the whole sample",
        "warning": "SYNTHETIC DATA - demonstrates software behaviour, not market alpha",
    }
    (d / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ohlcv": f"{name}.csv", "meta": f"{name}.meta.json"}


def load_dataset(directory: str | Path, name: str) -> dict[str, Any]:
    """Load a dataset written by :func:`write_synthetic_dataset` or the market downloader."""
    d = Path(directory)
    p = d / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; generate or download it first")
    ohlcv = pd.read_csv(p, parse_dates=["date"])
    uni = pd.read_csv(d / f"{name}_universe.csv")
    bench_p = d / f"{name}_benchmark.csv"
    bench = None
    if bench_p.exists():
        b = pd.read_csv(bench_p, index_col=0, parse_dates=True)
        bench = b.iloc[:, 0]
    meta_p = d / f"{name}.meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    out = {"ohlcv": ohlcv, "universe": uni, "benchmark": bench, "meta": meta}
    for extra in ("truth", "pair_truth"):
        q = d / f"{name}_{extra}.csv"
        if q.exists():
            out[extra] = pd.read_csv(q)
    return out


def log_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Natural log of a price frame, keeping index/columns (typed as a DataFrame)."""
    return pd.DataFrame(np.log(df.to_numpy(dtype=float)), index=df.index, columns=df.columns)


def wide(ohlcv: pd.DataFrame, field_name: str = "close") -> pd.DataFrame:
    """Wide (date x symbol) frame of one field; missing bars stay NaN."""
    if field_name not in FIELDS:
        raise KeyError(field_name)
    return ohlcv.pivot(index="date", columns="symbol", values=field_name).sort_index()
