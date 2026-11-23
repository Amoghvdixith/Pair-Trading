"""Map an estimated spread relationship to actual share holdings, and enforce limits.

Log-price spread  s = log P_y - alpha - beta log P_x.
A small change is ds ~= r_y - beta r_x (r = simple returns, to first order).  Holding
dollar notionals N_y and N_x earns N_y r_y + N_x r_x, which is proportional to ds when

    N_x = -beta N_y          (beta-weighted notional hedge)

With gross exposure G = |N_y| + |N_x| and direction d (+1 = long spread):

    N_y = d G / (1 + |beta|),   N_x = -d beta G / (1 + |beta|)

This hedge is dollar-neutral ONLY if beta = 1; its net dollar exposure is
N_y + N_x = d G (1 - beta) / (1 + beta).  It is market-beta-neutral only if
beta_mkt(y) = beta * beta_mkt(x), which nothing guarantees.  Beta is NOT a share ratio:
shares are N / P, so the share ratio is beta * P_y / P_x.  The fitted intercept alpha is
a level offset, not a position, and is never traded.

For RAW-price spreads (s = P_y - alpha - beta P_x) beta IS a share ratio: hold q_y shares
of y and -beta q_y shares of x.

Shares are rounded toward zero; intended vs realised notionals are both reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LegTargets:
    y_shares: int
    x_shares: int
    y_notional_intended: float
    x_notional_intended: float
    y_notional: float
    x_notional: float
    beta: float
    rule: str

    @property
    def gross(self) -> float:
        return abs(self.y_notional) + abs(self.x_notional)

    @property
    def net(self) -> float:
        return self.y_notional + self.x_notional

    @property
    def dollar_imbalance(self) -> float:
        """|long value - short value| / gross (0 = dollar neutral)."""
        g = self.gross
        return abs(self.net) / g if g > 0 else math.nan

    @property
    def rounding_gap(self) -> float:
        """Realised minus intended gross, fraction of intended gross."""
        gi = abs(self.y_notional_intended) + abs(self.x_notional_intended)
        return (self.gross - gi) / gi if gi > 0 else math.nan


def beta_allowed(beta: float, beta_min: float, beta_max: float) -> bool:
    """Predeclared policy: trade only finite beta in [beta_min, beta_max] (beta_min > 0)."""
    return math.isfinite(beta) and beta_min <= beta <= beta_max


def notional_split(beta: float, gross: float, direction: int,
                   rule: str = "beta_weighted") -> tuple[float, float]:
    """Dollar notionals (N_y, N_x) for gross ``gross`` and direction +1/-1."""
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (long spread) or -1 (short spread)")
    if gross < 0:
        raise ValueError("gross must be >= 0")
    if rule == "beta_weighted":
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta_weighted rule needs a positive finite beta")
        k = gross / (1.0 + beta)
        return direction * k, -direction * beta * k
    if rule == "dollar_neutral":
        return direction * gross / 2.0, -direction * gross / 2.0
    raise ValueError(f"unknown notional rule {rule!r}")


def shares_toward_zero(notional: float, price: float) -> int:
    if not (math.isfinite(price) and price > 0):
        raise ValueError("price must be positive and finite")
    return int(math.trunc(notional / price))


def pair_holdings(beta: float, gross: float, direction: int, price_y: float, price_x: float,
                  rule: str = "beta_weighted") -> LegTargets:
    """Whole-share holdings for a log-price spread hedge."""
    ny, nx = notional_split(beta, gross, direction, rule)
    qy, qx = shares_toward_zero(ny, price_y), shares_toward_zero(nx, price_x)
    return LegTargets(qy, qx, ny, nx, qy * price_y, qx * price_x, beta, rule)


def raw_price_holdings(beta: float, y_shares: int) -> tuple[int, int]:
    """Raw-price spread: beta is a share ratio (q_x = -beta q_y), rounded toward zero."""
    return int(y_shares), int(math.trunc(-beta * y_shares))


def unit_pnl_vol(ret_y: pd.Series, ret_x: pd.Series, beta: float,
                 rule: str = "beta_weighted") -> float:
    """Daily sd of P&L per 1 unit of gross for the hedged position (formation data).

    This puts pairs on a comparable P&L-risk scale (fraction of gross), unlike raw
    residual standard deviations from different models.
    """
    wy, wx = notional_split(beta, 1.0, 1, rule)
    pnl = (wy * ret_y + wx * ret_x).dropna()
    return float(pnl.std(ddof=1)) if len(pnl) > 2 else math.nan


def inverse_vol_gross(unit_vols: dict[str, float], budget: float, cap: float,
                      vol_floor: float = 1e-3) -> dict[str, float]:
    """Split a gross budget across pairs in proportion to 1 / unit P&L vol (floored).

    Equal marginal pair volatility is NOT equal portfolio risk when pairs are correlated
    (e.g. share a stock or a sector factor).  Each pair is also capped at ``cap``.
    """
    if not unit_vols:
        return {}
    inv = {k: 1.0 / max(v if math.isfinite(v) else vol_floor, vol_floor)
           for k, v in unit_vols.items()}
    tot = sum(inv.values())
    return {k: min(cap, budget * v / tot) for k, v in inv.items()}


@dataclass(frozen=True)
class Limits:
    portfolio_gross_cap: float = 1.0
    net_cap: float = 0.2
    name_cap: float = 0.15
    sector_gross_cap: float = 0.5
    max_pairs_per_name: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Limits:
        d = dict(d or {})
        return cls(**{k: (int(v) if k == "max_pairs_per_name" else float(v))
                      for k, v in d.items()})


def check_book(weights: dict[str, float], sectors: dict[str, str], lim: Limits,
               tol: float = 1e-9) -> list[str]:
    """Return a list of violated limits for a stock-level weight book."""
    out = []
    gross = sum(abs(v) for v in weights.values())
    if gross > lim.portfolio_gross_cap + tol:
        out.append(f"gross {gross:.3f} > {lim.portfolio_gross_cap}")
    net = sum(weights.values())
    if abs(net) > lim.net_cap + tol:
        out.append(f"net {net:.3f} beyond {lim.net_cap}")
    for s, w in weights.items():
        if abs(w) > lim.name_cap + tol:
            out.append(f"name {s} {w:.3f} beyond {lim.name_cap}")
    sec: dict[str, float] = {}
    for s, w in weights.items():
        sec[sectors.get(s, "?")] = sec.get(sectors.get(s, "?"), 0.0) + abs(w)
    for k, v in sec.items():
        if v > lim.sector_gross_cap + tol:
            out.append(f"sector {k} {v:.3f} > {lim.sector_gross_cap}")
    return out


def admit_scale(current: dict[str, float], new: dict[str, float], sectors: dict[str, str],
                lim: Limits, min_scale: float = 0.25) -> tuple[float, str]:
    """Largest k in [min_scale, 1] such that current + k*new satisfies all limits.

    Returns (0, reason) if even ``min_scale`` violates a limit.  Existing positions are
    never resized by admission control.  Bisection on k (constraints are monotone-ish in
    k; the final k is re-verified).
    """
    def book(k: float) -> dict[str, float]:
        b = dict(current)
        for s, w in new.items():
            b[s] = b.get(s, 0.0) + k * w
        return b

    if not check_book(book(1.0), sectors, lim):
        return 1.0, ""
    lo_viol = check_book(book(min_scale), sectors, lim)
    if lo_viol:
        return 0.0, "; ".join(lo_viol)
    lo, hi = min_scale, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if check_book(book(mid), sectors, lim):
            hi = mid
        else:
            lo = mid
    return lo, "scaled to satisfy limits"


def weights_from_shares(shares: dict[str, float], prices: dict[str, float],
                        equity: float) -> dict[str, float]:
    if equity <= 0:
        raise ValueError("equity must be positive")
    return {s: q * prices[s] / equity for s, q in shares.items() if q != 0}


def engine_weight_for_shares(q: int, price: float, equity: float) -> float:
    """Weight that makes the backtester's ``trunc(w * equity / price)`` return exactly q.

    The engine sizes orders as trunc(w * equity / price).  Passing (|q| + 0.5) * sign(q)
    shares' worth of weight avoids floating-point truncation (q - 1e-12 -> q - 1).
    """
    if q == 0:
        return 0.0
    return float(np.sign(q)) * (abs(q) + 0.5) * price / equity
