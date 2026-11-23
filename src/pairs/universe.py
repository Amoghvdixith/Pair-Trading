"""Formation-time universe filters and candidate-pair generation.

Every filter uses ONLY the formation window ``[formation_start, cutoff]``: no future
liquidity, no future data completeness.  Rejected symbols are recorded with a reason.

Missing-data rules (explicit)
-----------------------------
* history: observed closes / formation sessions >= ``min_history_frac``
* gaps: longest run of missing sessions <= ``max_consecutive_missing``
* recency: last observation within ``require_last_obs_within`` sessions of the cutoff
* liquidity: median(close * volume) over observed formation sessions >= threshold
* currency: must equal the configured single currency (no silent FX mixing; GBp vs GBP
  is a different price unit and is rejected unless converted explicitly upstream)

Point-in-time membership is optional: a membership table with ``known_from`` /
``known_to`` columns restricts candidates to symbols that were members AT the cutoff.
Without it the universe is the static list in the metadata file, which is NOT a
historical point-in-time universe (survivorship bias - documented in docs/limitations.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseRules:
    min_history_frac: float = 0.95
    max_consecutive_missing: int = 5
    require_last_obs_within: int = 3
    min_median_dollar_volume: float = 5.0e6
    currency: str = "USD"
    candidate_rule: str = "same_industry"

    def __post_init__(self) -> None:
        if not 0 < self.min_history_frac <= 1:
            raise ValueError("min_history_frac must be in (0, 1]")
        if self.candidate_rule not in ("same_industry", "same_sector"):
            raise ValueError("candidate_rule must be same_industry or same_sector")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> UniverseRules:
        # YAML 1.1 reads "5.0e6" as a string: coerce numeric fields explicitly
        d = dict(d or {})
        for k in ("min_history_frac", "min_median_dollar_volume"):
            if k in d:
                d[k] = float(d[k])
        for k in ("max_consecutive_missing", "require_last_obs_within"):
            if k in d:
                d[k] = int(d[k])
        return cls(**d)


def _longest_nan_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def filter_universe(close: pd.DataFrame, volume: pd.DataFrame, meta: pd.DataFrame,
                    cutoff: pd.Timestamp, rules: UniverseRules,
                    membership: pd.DataFrame | None = None) -> tuple[list[str], pd.DataFrame]:
    """Apply formation-time filters.

    ``close``/``volume`` must already be restricted to the formation window (the caller
    slices; this function asserts that no date is after ``cutoff``).
    Returns (eligible symbols, audit table with one row per symbol).
    """
    cutoff = pd.Timestamp(cutoff)
    if len(close.index) and close.index.max() > cutoff:
        raise ValueError("filter_universe received data after the formation cutoff")
    m = meta.set_index("symbol")
    n = len(close.index)
    rows = []
    for sym in close.columns:
        reason = ""
        c = close[sym].to_numpy(dtype=float)
        obs = np.isfinite(c)
        frac = float(obs.mean()) if n else 0.0
        if sym not in m.index:
            reason = "no metadata"
        elif membership is not None and not _member_at(membership, sym, cutoff):
            reason = "not a member at cutoff (point-in-time)"
        elif str(m.loc[sym, "currency"]) != rules.currency:
            reason = f"currency {m.loc[sym, 'currency']} != {rules.currency}"
        elif frac < rules.min_history_frac:
            reason = f"history {frac:.2%} < {rules.min_history_frac:.0%}"
        elif _longest_nan_run(~obs) > rules.max_consecutive_missing:
            reason = f"missing run {_longest_nan_run(~obs)} > {rules.max_consecutive_missing}"
        elif n - 1 - int(np.flatnonzero(obs)[-1]) > rules.require_last_obs_within:
            reason = "stale: no recent observation before cutoff"
        else:
            dv = (close[sym] * volume[sym]).dropna()
            med = float(dv.median()) if len(dv) else 0.0
            if med < rules.min_median_dollar_volume:
                reason = f"median $vol {med:,.0f} < {rules.min_median_dollar_volume:,.0f}"
        rows.append({"symbol": sym, "eligible": reason == "", "reason": reason,
                     "history_frac": frac})
    audit = pd.DataFrame(rows)
    elig = audit.loc[audit["eligible"], "symbol"].tolist()
    return elig, audit


def _member_at(membership: pd.DataFrame, sym: str, date: pd.Timestamp) -> bool:
    g = membership[membership["symbol"] == sym]
    if g.empty:
        return False
    kf = pd.to_datetime(g["known_from"])
    kt = pd.to_datetime(g["known_to"])
    return bool(((kf <= date) & (kt.isna() | (date <= kt))).any())


def candidate_pairs(symbols: list[str], meta: pd.DataFrame,
                    rule: str = "same_industry") -> pd.DataFrame:
    """All unordered same-industry (or same-sector) pairs, sorted deterministically."""
    col = "industry" if rule == "same_industry" else "sector"
    m = meta.set_index("symbol")
    out = []
    groups: dict[str, list[str]] = {}
    for s in sorted(symbols):
        groups.setdefault(str(m.loc[s, col]), []).append(s)
    for g, syms in sorted(groups.items()):
        for a, b in combinations(sorted(syms), 2):
            out.append({"pair": f"{a}-{b}", "a": a, "b": b, "group": g,
                        "sector": str(m.loc[a, "sector"])})
    return pd.DataFrame(out, columns=["pair", "a", "b", "group", "sector"])
