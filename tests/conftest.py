"""Shared offline fixtures (no network, explicit seeds)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pairs import data as D
from pairs.cohorts import Cohort
from pairs.hedge import ols_hedge
from pairs.screening import ScreenConfig, ScreenResult
from pairs.strategy import CohortPlan, PairSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def screen_raw() -> dict:
    return yaml.safe_load((ROOT / "configs" / "screening.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def strat_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "strategy.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def screen_cfg(screen_raw: dict) -> ScreenConfig:
    return ScreenConfig.from_dict(screen_raw)


def pair_market(n: int = 520, seed: int = 7, half_life: float = 5.0, noise_sd: float = 0.04,
                drop: dict[str, list[int]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-stock cointegrated OHLCV universe ("AAA" = y, "BBB" = x)."""
    df, _ = D.cointegrated_pair(n, alpha=0.3, beta=1.2, half_life=half_life,
                                noise_sd=noise_sd, x_start=np.log(50.0), seed=seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    rng = np.random.default_rng(seed)
    rows = []
    for sym, col in (("AAA", "y"), ("BBB", "x")):
        bars = pd.DataFrame(D.ohlcv_from_close(df[col].to_numpy(), rng, 5e7), index=idx)
        bars.insert(0, "symbol", sym)
        bars.index.name = "date"
        b = bars.reset_index()
        if drop and sym in drop:
            b = b.drop(index=drop[sym])
        rows.append(b)
    oh = pd.concat(rows, ignore_index=True)
    meta = pd.DataFrame({"symbol": ["AAA", "BBB"], "sector": ["S", "S"],
                         "industry": ["S", "S"], "currency": ["USD", "USD"]})
    return oh, meta


def manual_plan(oh: pd.DataFrame, formation_end: int = 260, trade_len: int = 250,
                gross: float = 0.3, cohort_id: int = 0, share: float = 1.0) -> CohortPlan:
    close = D.wide(oh, "close")
    idx = close.index
    c = Cohort(cohort_id, idx[0], idx[formation_end - 1], idx[formation_end],
               idx[min(formation_end + trade_len, len(idx)) - 1], share)
    lp = np.log(close.loc[: c.cutoff])
    fit = ols_hedge(lp["AAA"], lp["BBB"])
    spec = PairSpec(cohort_id, "AAA-BBB", "AAA", "BBB", "S", fit.alpha, fit.beta, 5.0,
                    float(fit.resid.std()), 0.001, 0.001, 0.01, gross, 1e-5,
                    fit.resid_sd ** 2, (fit.alpha, fit.beta), (1.0, 1.0))
    empty = pd.DataFrame()
    return CohortPlan(c, ScreenResult(c.cutoff, c.formation_start, empty, empty, empty, {}),
                      [spec])


@pytest.fixture()
def pair_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pair_market()
