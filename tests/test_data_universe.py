from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs import data as D
from pairs.universe import UniverseRules, candidate_pairs, filter_universe


def test_synthetic_reproducible_and_seed_sensitive() -> None:
    a1, _ = D.cointegrated_pair(300, seed=3)
    a2, _ = D.cointegrated_pair(300, seed=3)
    a3, _ = D.cointegrated_pair(300, seed=4)
    pd.testing.assert_frame_equal(a1, a2)
    assert not np.allclose(a1["y"], a3["y"])
    spec = D.UniverseSpec(start="2015-01-01", end="2016-06-30", n_sectors=2,
                          stocks_per_sector=4, seed=11)
    u1, u2 = D.generate_universe(spec), D.generate_universe(spec)
    pd.testing.assert_frame_equal(u1["ohlcv"], u2["ohlcv"])


def test_ohlcv_is_coherent() -> None:
    rng = np.random.default_rng(0)
    lc = np.cumsum(rng.normal(0, 0.01, 400)) + np.log(30)
    b = D.ohlcv_from_close(lc, rng)
    assert np.all(b["high"] >= np.maximum(b["open"], b["close"]) - 1e-12)
    assert np.all(b["low"] <= np.minimum(b["open"], b["close"]) + 1e-12)
    assert np.all(b["volume"] > 0)
    np.testing.assert_allclose(np.log(b["close"]), lc)
    # the open is not the same-day close (it is a bridge point, gaps exist)
    assert np.mean(np.isclose(b["open"][1:], b["close"][1:])) < 0.01


def test_correlated_rw_requires_nonsingular_covariance() -> None:
    with pytest.raises(ValueError):
        D.correlated_random_walks(100, rho=1.0)
    df, _ = D.correlated_random_walks(3000, rho=0.8, seed=1)
    assert 0.75 < df["y"].diff().corr(df["x"].diff()) < 0.85


def test_pair_truth_marks_linked_pairs() -> None:
    spec = D.UniverseSpec(start="2015-01-01", end="2016-12-30", n_sectors=2,
                          stocks_per_sector=6, seed=5)
    u = D.generate_universe(spec)
    pt = u["pair_truth"]
    # null sector (last) has no cointegrated pairs; first sector has C(4,2)=6 linked pairs
    assert pt[pt["sector"] == spec.sector_names[1]]["cointegrated"].sum() == 0
    assert pt[pt["sector"] == spec.sector_names[0]]["cointegrated"].sum() == 6


def _frames(n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0}, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    vol["D"] = 10.0  # illiquid
    close.loc[idx[10:20], "B"] = np.nan  # 10-session gap
    close.loc[idx[-4:], "C"] = np.nan  # stale (gap 4 <= 5, but last obs 4 sessions old)
    meta = pd.DataFrame({"symbol": list("ABCDE"), "sector": ["X"] * 5, "industry": ["X"] * 5,
                         "currency": ["USD", "USD", "USD", "USD", "GBP"]})
    return close, vol, meta


def test_universe_filters_record_reasons() -> None:
    close, vol, meta = _frames()
    rules = UniverseRules(min_history_frac=0.85, max_consecutive_missing=5,
                          require_last_obs_within=3, min_median_dollar_volume=1e5)
    elig, audit = filter_universe(close, vol, meta, close.index[-1], rules)
    assert elig == ["A"]
    r = audit.set_index("symbol")["reason"]
    assert "missing run" in r["B"]
    assert "stale" in r["C"]
    assert "median $vol" in r["D"]
    assert "currency" in r["E"]


def test_universe_filter_rejects_future_data() -> None:
    close, vol, meta = _frames()
    with pytest.raises(ValueError, match="after the formation cutoff"):
        filter_universe(close, vol, meta, close.index[50], UniverseRules())


def test_point_in_time_membership() -> None:
    close, vol, meta = _frames()
    close["B"] = 20.0
    mem = pd.DataFrame({"symbol": ["A", "B"], "known_from": ["2019-01-01", "2030-01-01"],
                        "known_to": [None, None]})
    elig, audit = filter_universe(close, vol, meta, close.index[-1],
                                  UniverseRules(min_median_dollar_volume=1e5), mem)
    assert "A" in elig and "B" not in elig


def test_candidate_pairs_same_industry_only() -> None:
    meta = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "sector": ["X", "X", "Y", "Y"],
                         "industry": ["x1", "x1", "y1", "y2"], "currency": ["USD"] * 4})
    c = candidate_pairs(["D", "C", "B", "A"], meta)
    assert c["pair"].tolist() == ["A-B"]
    c2 = candidate_pairs(["A", "B", "C", "D"], meta, "same_sector")
    assert set(c2["pair"]) == {"A-B", "C-D"}
