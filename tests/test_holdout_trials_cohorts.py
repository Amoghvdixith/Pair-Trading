from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from backtester.walkforward import HoldoutViolation, TrialRegistry, check_holdout_lock

from pairs.cohorts import make_schedule
from pairs.experiments import HoldoutNotConfirmed, build_context, holdout_evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_schedule_boundaries() -> None:
    idx = pd.bdate_range("2012-01-02", "2016-12-30")
    coh = make_schedule(idx, 12, 6, 3, first_trade_month="2013-01", last_trade_end="2016-12-31")
    c0 = coh[0]
    assert str(c0.cutoff.date()) == "2012-12-31" and str(c0.trade_start.date()) == "2013-01-01"
    assert str(c0.trade_end.date()) == "2013-06-28"
    for c in coh:
        assert c.formation_start < c.cutoff < c.trade_start <= c.trade_end
        assert c.trade_end <= pd.Timestamp("2016-12-31")
    assert coh[-1].trade_end == pd.Timestamp("2016-12-30")
    assert all(c.capital_share == 0.5 for c in coh)
    monthly = make_schedule(idx, 12, 6, 1)
    assert monthly[0].capital_share == pytest.approx(1 / 6)


def _tmp_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "data" / "synthetic").mkdir(parents=True)
    shutil.copy(ROOT / "pyproject.toml", root)
    shutil.copytree(ROOT / "configs", root / "configs")
    return root


def test_holdout_requires_confirmation_and_is_excluded(tmp_path: Path) -> None:
    root = _tmp_project(tmp_path)
    # small dataset to keep the test fast
    exp = (root / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    exp = exp.replace("fast:   {n_sectors: 4, stocks_per_sector: 6}",
                      "fast:   {n_sectors: 2, stocks_per_sector: 4}")
    (root / "configs" / "experiments.yaml").write_text(exp, encoding="utf-8")
    ctx = build_context(root, "synthetic", "fast")
    assert ctx.ohlcv["date"].max() < pd.Timestamp(ctx.split["holdout_start"])
    lock = json.loads((root / "results" / "logs" / "holdout_lock.json").read_text())
    assert lock[ctx.dataset]["holdout_start"] == ctx.split["holdout_start"]
    with pytest.raises(HoldoutNotConfirmed):
        holdout_evaluation(root, "synthetic", "fast", confirm=False)
    assert not (root / "results" / "logs" / "holdout_evaluations.jsonl").exists()
    with pytest.raises(HoldoutViolation):
        check_holdout_lock(root / "results" / "logs" / "holdout_lock.json", "2021-01-04",
                           ctx.dataset)


def test_trial_registry_appends(tmp_path: Path) -> None:
    reg = TrialRegistry(tmp_path / "t.jsonl")
    reg.append(pd.DataFrame([{"sample": "validation", "entry_z": 2.0, "sharpe": 0.1}]))
    reg.append(pd.DataFrame([{"sample": "test", "entry_z": 2.0, "sharpe": -0.1}]))
    assert len(reg.read()) == 2
