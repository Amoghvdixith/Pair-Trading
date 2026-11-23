"""Integration with the project-02 event-driven backtester (``backtester==0.1.0``).

Public backtester interfaces used: ``Backtest``, ``BacktestConfig``, ``MarketData``,
``Strategy``, ``Sizer``, ``PortfolioConstraints``, ``CostModel`` (+ ``Commission``,
``SpreadCost``, ``Slippage``), ``FinancingModel``, ``ExecutionConfig``,
``BacktestResult``, ``compute_metrics`` and ``backtester.walkforward`` hold-out helpers.

One non-API coupling: the adapter reads ``Backtest.portfolio`` (a public attribute set in
``Backtest.run``) through a callable, to size share targets from current equity and to
confirm two-leg fills.  See docs/backtester_integration.md (change request CR-1).

Execution conventions inherited from the backtester: signal at the close of t, fill at
the next tradable open, sells before buys, whole shares, costs itemised per fill,
financing accrued daily on cash / short balances, accounting reconciled every close.
Pair fills are NOT atomic: each leg is an independent order.  With ``expires_after=0`` a
leg that cannot trade at its first eligible open is cancelled; the strategy then unwinds
any filled leg at the next open (deterministic legging policy).
"""

from __future__ import annotations

import importlib.metadata as md
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from backtester import (
    Backtest,
    BacktestConfig,
    BacktestResult,
    Commission,
    CostModel,
    ExecutionConfig,
    FinancingModel,
    MarketData,
    PointInTimeView,
    PortfolioConstraints,
    SignalEvent,
    Sizer,
    Slippage,
    SpreadCost,
)

from .strategy import PairsStrategy


class PassThroughSizer(Sizer):
    """Use each signal's ``direction`` as the target weight (the strategy already sized)."""

    def weights(self, signals: list[SignalEvent], view: PointInTimeView) -> dict[str, float]:
        return {s.symbol: float(s.direction) for s in signals}


def backtester_version() -> str:
    try:
        return md.version("backtester")
    except md.PackageNotFoundError:  # pragma: no cover
        return "not installed"


def market_data_from_long(ohlcv: pd.DataFrame, symbols: list[str] | None = None,
                          provenance: dict[str, Any] | None = None) -> MarketData:
    df = ohlcv if symbols is None else ohlcv[ohlcv["symbol"].isin(symbols)]
    return MarketData.from_long(df[["date", "symbol", "open", "high", "low", "close",
                                    "volume"]], provenance)


@dataclass(frozen=True)
class RunSpec:
    """Cost / financing scenario for one run.

    scenario: ``gross`` (no transaction costs, no borrow, no interest),
              ``net`` (detailed costs + borrow + financing), or
              ``flat`` (flat one-way bps per leg REPLACING commission/spread/slippage;
              borrow and financing unchanged from ``net``).
    """

    scenario: str = "net"
    flat_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.scenario not in ("gross", "net", "flat"):
            raise ValueError("scenario must be gross, net or flat")


def cost_model(strat_cfg: dict[str, Any], spec: RunSpec) -> CostModel:
    if spec.scenario == "gross":
        return CostModel.zero()
    if spec.scenario == "flat":
        return CostModel.proportional(spec.flat_bps)
    c = strat_cfg["costs"]
    return CostModel(commission=Commission(rate_bps=float(c["commission_bps"])),
                     spread=SpreadCost(float(c["spread_bps_full"])),
                     slippage=Slippage(float(c["slippage_bps"])))


def financing_model(strat_cfg: dict[str, Any], spec: RunSpec) -> FinancingModel:
    if spec.scenario == "gross":
        return FinancingModel()
    f = strat_cfg["financing"]
    return FinancingModel(credit_rate=float(f["credit_rate"]), debit_rate=float(f["debit_rate"]),
                          borrow_fee=float(f["borrow_fee"]),
                          short_proceeds_earn_interest=bool(f["short_proceeds_earn_interest"]))


def run_pairs_backtest(data: MarketData, strategy: PairsStrategy, strat_cfg: dict[str, Any],
                       spec: RunSpec, trade_start: pd.Timestamp, label: str,
                       data_mode: str) -> BacktestResult:
    """Run one PairsStrategy through the backtester and attach pair-level logs."""
    lim = strategy.p.limits
    # Backstop only: the strategy enforces limits itself; a binding backstop would
    # rescale every weight and break hedges, so it is deliberately loose (and logged).
    constraints = PortfolioConstraints(max_gross=lim.portfolio_gross_cap * 1.25,
                                       max_net=1.0, max_position=min(1.0, lim.name_cap * 2),
                                       allow_short=True)
    bt = Backtest(
        data, strategy, sizer=PassThroughSizer(), constraints=constraints,
        costs=cost_model(strat_cfg, spec),
        execution=ExecutionConfig(expires_after=int(strat_cfg["execution"]["expires_after"])),
        financing=financing_model(strat_cfg, spec),
        config=BacktestConfig(initial_cash=float(strat_cfg["initial_cash"]), fractional=False,
                              trade_start=trade_start, liquidate_on_final_open=True),
        label=label, data_mode=data_mode)
    strategy.attach_portfolio(lambda: bt.portfolio)
    res = bt.run()
    res.meta["pairs"] = {
        "scenario": spec.scenario, "flat_bps": spec.flat_bps,
        "backtester_version": backtester_version(),
        "hedge_method": strategy.p.method, "notional_rule": strategy.p.notional_rule,
        "pair_fill_model": "non-atomic legs; expires_after=0; unwind on leg failure",
        "n_cohorts": len(strategy.plans),
        "n_pairs_planned": int(sum(len(p.pairs) for p in strategy.plans)),
    }
    res.meta["backstop_bound"] = _backstop_bound(res, strategy)
    return res


def _backstop_bound(res: BacktestResult, strategy: PairsStrategy) -> bool:
    """True if the engine's targets ever differ in sign/size from the emitted weights."""
    if res.signals.empty or res.targets.empty:
        return False
    s = res.signals.set_index(["timestamp", "symbol"])["direction"]
    t = res.targets.set_index(["date", "symbol"])["target_weight"]
    t.index = t.index.set_names(["timestamp", "symbol"])
    both = pd.concat([s, t], axis=1, join="inner").dropna()
    return bool((np.abs(both.iloc[:, 0] - both.iloc[:, 1]) > 1e-12).any())
