"""Rolling formation / trading cohorts.

Cohort k:  formation = sessions in (cutoff_k - F months, cutoff_k],
           trading   = sessions in (cutoff_k, cutoff_k + T months]
with cutoffs at month ends spaced ``step_months`` apart.

Capital: ONE shared account.  Each cohort may use ``capital_share`` of the portfolio
gross budget, with capital_share = 1 / ceil(T / step) = the maximum number of cohorts
that can trade simultaneously.  Overlapping cohorts therefore never receive independent
full copies of the account, and returns are measured once, on the single account.

Decision timing inside a trading window [first, last]:
  * signals are evaluated at closes first .. last-1;
  * at the close of ``last - 1`` the cohort is marked expiring: all its pairs are sent
    flat and fill at the open of ``last`` (inside the window, with costs);
  * no entries are taken on the expiring close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Cohort:
    cohort_id: int
    formation_start: pd.Timestamp
    cutoff: pd.Timestamp
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp
    capital_share: float

    def as_dict(self) -> dict[str, object]:
        return {k: (str(v.date()) if isinstance(v, pd.Timestamp) else v)
                for k, v in self.__dict__.items()}


def make_schedule(index: pd.DatetimeIndex, formation_months: int = 12, trading_months: int = 6,
                  step_months: int = 3, first_trade_month: str | None = None,
                  last_trade_end: str | pd.Timestamp | None = None) -> list[Cohort]:
    """Build cohorts whose formation AND trading windows lie inside ``index``.

    ``first_trade_month`` (YYYY-MM) sets the month in which the first trading window
    starts; ``last_trade_end`` drops cohorts whose trading window would end after it.
    """
    if formation_months < 1 or trading_months < 1 or step_months < 1:
        raise ValueError("window lengths must be positive")
    idx = pd.DatetimeIndex(index).sort_values()
    if len(idx) == 0:
        return []
    share = 1.0 / math.ceil(trading_months / step_months)
    month_end = idx.to_series().groupby(idx.to_period("M")).max()
    if first_trade_month is not None:
        p0 = pd.Period(first_trade_month, "M") - 1
    else:
        p0 = idx[0].to_period("M") + formation_months - 1
    lim = pd.Timestamp(last_trade_end) if last_trade_end is not None else idx[-1]
    out = []
    p = p0
    cid = 0
    while True:
        if p not in month_end.index:
            if p > idx[-1].to_period("M"):
                break
            p += step_months
            continue
        cutoff = month_end[p]
        f_lo = (p - formation_months + 1).start_time
        t_hi_p = p + trading_months
        if t_hi_p not in month_end.index:
            break
        t_end = month_end[t_hi_p]
        if t_end > lim:
            break
        form = idx[(idx >= f_lo) & (idx <= cutoff)]
        trade = idx[(idx > cutoff) & (idx <= t_end)]
        if (p - formation_months + 1) >= idx[0].to_period("M") and len(form) and len(trade) >= 3:
            out.append(Cohort(cid, form[0], cutoff, trade[0], trade[-1], share))
            cid += 1
        p += step_months
    return out


def overlap_count(cohorts: list[Cohort], date: pd.Timestamp) -> int:
    d = pd.Timestamp(date)
    return sum(1 for c in cohorts if c.trade_start <= d <= c.trade_end)
