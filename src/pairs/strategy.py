"""Cohort plans (formation-only research) and the online PairsStrategy.

Separation of concerns
----------------------
* :func:`build_cohort_plan` - RESEARCH on formation data only: screening, OU fit,
  Kalman tuning and risk budgets.  Its input is sliced at the cutoff and asserted.
* :class:`PairsStrategy` - SIGNALS + target holdings.  It subclasses the project-02
  ``backtester.Strategy`` and sees prices only through the backtester's point-in-time
  view.  It emits exact share targets (as engine weights) for the whole stock book.
* The backtester - EXECUTION and ACCOUNTING: next-open fills, costs, financing,
  cash, FIFO lots, reconciliation.  Nothing here re-implements accounting.

Frozen within a cohort: the selected pairs, orientation, static coefficients,
half-life-based windows/time stops, Kalman Q/R/prior and gross budgets.
Causal updates allowed: rolling-OLS coefficients (window ending t-1), the Kalman state
(filter only), z-score windows, breakdown checks on trailing data.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from backtester import MarketEvent, SignalEvent, Strategy

from .allocation import (
    Limits,
    admit_scale,
    beta_allowed,
    engine_weight_for_shares,
    inverse_vol_gross,
    pair_holdings,
    unit_pnl_vol,
)
from .cohorts import Cohort
from .data import log_prices
from .hedge import KalmanHedge, delta_to_Q, ols_hedge, tune_kalman
from .screening import ScreenConfig, ScreenResult, engle_granger, screen_pairs
from .signals import (
    Action,
    PairState,
    PairStateMachine,
    SignalConfig,
    max_attainable_z,
    zscore_last,
)


# --------------------------------------------------------------------------- plans
@dataclass(frozen=True)
class PairSpec:
    cohort_id: int
    pair: str
    y: str
    x: str
    sector: str
    alpha: float
    beta: float
    half_life: float
    spread_sd: float
    eg_pvalue: float
    p_adj: float
    unit_vol: float
    gross_frac: float
    kalman_delta: float
    kalman_R: float
    kalman_m0: tuple[float, float]
    kalman_P0: tuple[float, float]

    @property
    def key(self) -> str:
        return f"{self.cohort_id}:{self.pair}"


@dataclass
class CohortPlan:
    cohort: Cohort
    screen: ScreenResult
    pairs: list[PairSpec]


def build_cohort_plan(close: pd.DataFrame, volume: pd.DataFrame, meta: pd.DataFrame,
                      cohort: Cohort, screen_cfg: ScreenConfig, strat_cfg: dict[str, Any],
                      membership: pd.DataFrame | None = None) -> CohortPlan:
    """Formation-only research for one cohort (``close``/``volume`` may extend further;
    they are sliced here and the slice is asserted)."""
    sl = slice(cohort.formation_start, cohort.cutoff)
    c, v = close.loc[sl], volume.loc[sl]
    if c.index.max() > cohort.cutoff:
        raise AssertionError("formation slice leaks past the cutoff")
    res = screen_pairs(c, v, meta, cohort.cutoff, screen_cfg, membership)
    sizing, lim = strat_cfg["sizing"], Limits.from_dict(strat_cfg["constraints"])
    kcfg = strat_cfg["hedge"]["kalman"]
    rule = sizing.get("notional_rule", "beta_weighted")
    rets = c.pct_change(fill_method=None)
    lp = log_prices(c)
    tmp: list[dict[str, Any]] = []
    for r in res.selected.to_dict("records"):
        y, x = r["y"], r["x"]
        uv = unit_pnl_vol(rets[y], rets[x], r["beta"], rule)
        tune = tune_kalman(lp[y], lp[x], [float(d) for d in kcfg["delta_grid"]],
                           (float(kcfg["prior_var_alpha"]), float(kcfg["prior_var_beta"])))
        tmp.append({"r": r, "uv": uv, "tune": tune})
    budget = lim.portfolio_gross_cap * cohort.capital_share
    gross = inverse_vol_gross({t["r"]["pair"]: t["uv"] for t in tmp}, budget,
                              float(sizing["pair_gross_cap"]))
    pairs = []
    for t in tmp:
        r, tune = t["r"], t["tune"]
        g = gross[r["pair"]]
        if g < float(sizing["pair_gross_floor"]):
            continue
        pairs.append(PairSpec(
            cohort.cohort_id, r["pair"], r["y"], r["x"], r["sector"], float(r["alpha"]),
            float(r["beta"]), float(r["ou_half_life"]), float(r["spread_sd"]),
            float(r["p_pair"]), float(r["p_adj"]), float(t["uv"]), float(g),
            tune.delta, tune.R, tune.m0, tune.P0_diag))
    return CohortPlan(cohort, res, pairs)


def build_plans(close: pd.DataFrame, volume: pd.DataFrame, meta: pd.DataFrame,
                cohorts: list[Cohort], screen_cfg: ScreenConfig, strat_cfg: dict[str, Any],
                membership: pd.DataFrame | None = None) -> list[CohortPlan]:
    return [build_cohort_plan(close, volume, meta, c, screen_cfg, strat_cfg, membership)
            for c in cohorts]


# --------------------------------------------------------------------------- runtime
@dataclass
class _PairRT:
    spec: PairSpec
    cohort: Cohort
    sm: PairStateMachine
    window: int
    kf: KalmanHedge | None = None
    kf_last: pd.Timestamp | None = None
    last_z: float = math.nan
    entry_beta: float = math.nan
    direction: int = 0
    sessions: int = 0
    trade_id: int = -1
    pending: str = ""  # "" | entry | exit | rehedge
    prev_x_shares: int = 0


@dataclass
class StrategyParams:
    method: str = "static_ols"
    signal: SignalConfig = field(default_factory=SignalConfig)
    lookback_mult: float = 1.5
    lookback_min: int = 10
    lookback_max: int = 60
    time_stop_halflives: float = 3.0
    time_stop_min: int = 5
    time_stop_max: int = 120
    rolling_window: int = 60
    rehedge_threshold: float = 0.10
    notional_rule: str = "beta_weighted"
    beta_min: float = 0.2
    beta_max: float = 5.0
    breakdown_enabled: bool = True
    breakdown_every: int = 21
    breakdown_lookback: int = 252
    breakdown_pvalue: float = 0.20
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self) -> None:
        if self.method not in ("static_ols", "rolling_ols", "kalman"):
            raise ValueError("method must be static_ols, rolling_ols or kalman")
        zmax = max_attainable_z(self.lookback_min)
        if self.method == "static_ols" and self.signal.stop_z >= zmax:
            raise ValueError(
                f"stop_z={self.signal.stop_z} is unattainable: with lookback_min="
                f"{self.lookback_min} the rolling z-score is bounded by {zmax:.2f}")

    @classmethod
    def from_config(cls, strat_cfg: dict[str, Any], screen_cfg: ScreenConfig,
                    **overrides: Any) -> StrategyParams:
        s, h = strat_cfg["signals"], strat_cfg["hedge"]
        bd = s.get("breakdown", {})
        sig_kw = {k: overrides.pop(k) for k in ("entry_z", "exit_z", "stop_z", "entry_mode",
                                                 "cooldown", "max_missing") if k in overrides}
        sig = SignalConfig(entry_z=float(sig_kw.get("entry_z", s["entry_z"])),
                           exit_z=float(sig_kw.get("exit_z", s["exit_z"])),
                           stop_z=float(sig_kw.get("stop_z", s["stop_z"])),
                           entry_mode=sig_kw.get("entry_mode", s.get("entry_mode", "exceed")),
                           time_stop=1,  # per pair, set from half-life at activation
                           cooldown=int(sig_kw.get("cooldown", s.get("cooldown", 5))),
                           max_missing=int(sig_kw.get("max_missing", s.get("max_missing", 3))))
        p = cls(method=h["method"], signal=sig,
                lookback_mult=float(s["zscore_lookback_mult"]),
                lookback_min=int(s["zscore_lookback_min"]),
                lookback_max=int(s["zscore_lookback_max"]),
                time_stop_halflives=float(s["time_stop_halflives"]),
                time_stop_min=int(s["time_stop_min"]), time_stop_max=int(s["time_stop_max"]),
                rolling_window=int(h["rolling_window"]),
                rehedge_threshold=float(h["rehedge_threshold"]),
                notional_rule=strat_cfg["sizing"].get("notional_rule", "beta_weighted"),
                beta_min=screen_cfg.beta_min, beta_max=screen_cfg.beta_max,
                breakdown_enabled=bool(bd.get("enabled", True)),
                breakdown_every=int(bd.get("every", 21)),
                breakdown_lookback=int(bd.get("lookback", 252)),
                breakdown_pvalue=float(bd.get("pvalue", 0.2)),
                limits=Limits.from_dict(strat_cfg["constraints"]))
        for k, v in overrides.items():
            if not hasattr(p, k):
                raise ValueError(f"unknown strategy override {k!r}")
            setattr(p, k, v)
        p.__post_init__()
        return p


class PairsStrategy(Strategy):
    """Thin adapter: per-pair state machines -> exact stock share targets.

    The backtester gives strategies a point-in-time VIEW but no portfolio access; pair
    trading needs current equity (to size) and fill confirmation (two legs).  The adapter
    therefore receives a read-only ``portfolio_provider`` callable (the engine's
    ``Backtest.portfolio`` attribute).  This coupling is documented in
    docs/backtester_integration.md together with a change request for an official API.
    """

    name = "pairs"

    def __init__(self, plans: list[CohortPlan], params: StrategyParams,
                 sectors: dict[str, str], record_pairs: set[str] | None = None,
                 label: str = "") -> None:
        self.plans = plans
        self.p = params
        self.sectors = sectors
        self.record_pairs = record_pairs
        super().__init__(method=params.method, entry_z=params.signal.entry_z,
                         exit_z=params.signal.exit_z, stop_z=params.signal.stop_z,
                         lookback_mult=params.lookback_mult, label=label)
        self._portfolio_provider: Callable[[], Any] | None = None
        f_len = max((len(pd.bdate_range(pl.cohort.formation_start, pl.cohort.cutoff))
                     for pl in plans), default=260)
        self._nhist = max(f_len + 30, params.breakdown_lookback + 5,
                          params.rolling_window + 5, params.lookback_max + 5)

    def attach_portfolio(self, provider: Callable[[], Any]) -> None:
        self._portfolio_provider = provider

    def reset(self) -> None:
        self.rt: dict[str, _PairRT] = {}
        self.intended: dict[str, dict[str, int]] = {}
        self.confirmed: dict[str, dict[str, int]] = {}
        self.events: list[dict[str, Any]] = []
        self.ledger: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.series: list[dict[str, Any]] = []
        self.holdings_audit: list[dict[str, Any]] = []
        self._trade_seq = 0
        self._done: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _pf(self) -> Any:
        if self._portfolio_provider is None:
            raise RuntimeError("PairsStrategy needs attach_portfolio() before running")
        return self._portfolio_provider()

    def _log(self, now: pd.Timestamp, rt: _PairRT, kind: str, **kw: Any) -> None:
        self.events.append({"date": now, "cohort": rt.spec.cohort_id, "pair": rt.spec.pair,
                            "event": kind, "state": rt.sm.state.value, **kw})

    def _expected(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for book in self.intended.values():
            for s, q in book.items():
                out[s] = out.get(s, 0) + q
        return {s: q for s, q in out.items() if q != 0}

    def _record_ledger(self, now: pd.Timestamp, key: str, new: dict[str, int]) -> None:
        old = self.confirmed.get(key, {})
        for s in set(old) | set(new):
            d = new.get(s, 0) - old.get(s, 0)
            if d:
                self.ledger.append({"date": now, "key": key, "symbol": s, "delta": d})
        self.confirmed[key] = {s: q for s, q in new.items() if q}

    def _activate(self, spec: PairSpec, cohort: Cohort, hist: pd.DataFrame) -> _PairRT:
        p = self.p
        w = int(np.clip(round(p.lookback_mult * spec.half_life), p.lookback_min,
                        p.lookback_max))
        ts = int(np.clip(round(p.time_stop_halflives * spec.half_life), p.time_stop_min,
                         p.time_stop_max))
        sc = p.signal
        cfg = SignalConfig(sc.entry_z, sc.exit_z, sc.stop_z, sc.entry_mode, ts, sc.cooldown,
                           sc.max_missing)
        rt = _PairRT(spec, cohort, PairStateMachine(cfg), w)
        if p.method == "kalman":
            rt.kf = KalmanHedge(delta_to_Q(spec.kalman_delta), spec.kalman_R,
                                np.array(spec.kalman_m0), np.diag(spec.kalman_P0))
            form = hist.loc[cohort.formation_start:cohort.cutoff, [spec.y, spec.x]].dropna()
            for yy, xx in np.log(form.to_numpy()):
                rt.kf.step(float(yy), float(xx))
            rt.kf_last = form.index[-1] if len(form) else None
        return rt

    def _signal(self, rt: _PairRT, hist: pd.DataFrame) -> tuple[float, float, float, bool]:
        """Return (z, beta_now, spread_value, missing) at the current close."""
        sp, p = rt.spec, self.p
        today = hist.iloc[-1]
        py, px = today[sp.y], today[sp.x]
        if not (np.isfinite(py) and np.isfinite(px)):
            return math.nan, math.nan, math.nan, True
        if p.method == "static_ols":
            win = hist[[sp.y, sp.x]].iloc[-rt.window:]
            if win.isna().any().any():
                return math.nan, sp.beta, math.nan, False
            lg = np.log(win.to_numpy())
            s = lg[:, 0] - sp.alpha - sp.beta * lg[:, 1]
            return zscore_last(s), sp.beta, float(s[-1]), False
        if p.method == "rolling_ols":
            win = hist[[sp.y, sp.x]].iloc[-(p.rolling_window + 1):-1].dropna()
            if len(win) < p.rolling_window - 2:
                return math.nan, math.nan, math.nan, False
            lg = log_prices(win)
            try:
                fit = ols_hedge(lg[sp.y], lg[sp.x])
            except ValueError:
                return math.nan, math.nan, math.nan, False
            s = math.log(py) - fit.alpha - fit.beta * math.log(px)
            z = s / fit.resid_sd if fit.resid_sd > 1e-12 else math.nan
            return z, fit.beta, s, False
        # kalman: catch up on observations after the last processed date (causal)
        assert rt.kf is not None
        new = hist[[sp.y, sp.x]]
        if rt.kf_last is not None:
            new = new[new.index > rt.kf_last]
        new = new.dropna()
        st = None
        for t, (yy, xx) in zip(new.index, np.log(new.to_numpy()), strict=True):
            st = rt.kf.step(float(yy), float(xx))
            rt.kf_last = t
        if st is None or rt.kf_last != hist.index[-1]:
            return math.nan, float(rt.kf.m[1]), math.nan, False
        return st.z_innovation, st.beta, st.innovation, False

    def _breakdown(self, rt: _PairRT, hist: pd.DataFrame) -> bool:
        p = self.p
        if not p.breakdown_enabled or rt.sessions == 0 or rt.sessions % p.breakdown_every:
            return False
        win = log_prices(hist[[rt.spec.y, rt.spec.x]].iloc[-p.breakdown_lookback:].dropna())
        eg = engle_granger(win[rt.spec.y], win[rt.spec.x], min_obs=60)
        bd = eg.status == "ok" and eg.pvalue > p.breakdown_pvalue
        self.events.append({"date": hist.index[-1], "cohort": rt.spec.cohort_id,
                            "pair": rt.spec.pair, "event": "breakdown_check",
                            "state": rt.sm.state.value, "eg_pvalue": eg.pvalue,
                            "flag": bd})
        return bd

    def _price(self, s: str, today: pd.Series) -> float:
        v = float(today.get(s, np.nan))
        if np.isfinite(v):
            return v
        return float(self._pf().marks.get(s, np.nan))

    def _current_weights(self, today: pd.Series, equity: float) -> dict[str, float]:
        return {s: q * self._price(s, today) / equity for s, q in self._expected().items()}

    # ------------------------------------------------------------------ main hook
    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        now = self.view.now
        nxt = self.view.next_session()
        pf = self._pf()
        hist = self.view.history_frame("close", n=self._nhist)
        today = hist.iloc[-1]
        equity = pf.equity
        changed = False

        # 1) confirm or reject last close's orders (both legs must match exactly)
        expected = self._expected()
        actual = {s: int(round(q)) for s, q in pf.positions.items()}
        bad = {s for s in set(expected) | set(actual)
               if expected.get(s, 0) != actual.get(s, 0)}
        if bad:
            changed = True  # re-send the full book until positions match
        for key, rt in list(self.rt.items()):
            if not rt.pending:
                continue
            ok = not ({rt.spec.y, rt.spec.x} & bad)
            kind = rt.pending
            if kind == "entry":
                tag = rt.sm.confirm(ok)
                if ok:
                    self._record_ledger(now, key, self.intended[key])
                    self._trade_seq += 1
                    rt.trade_id = self._trade_seq
                    self.trades.append({"trade_id": rt.trade_id, "key": key,
                                        "cohort": rt.spec.cohort_id, "pair": rt.spec.pair,
                                        "side": rt.sm.state.value, "entry_fill": now,
                                        "entry_beta": rt.entry_beta,
                                        "y_shares": self.intended[key].get(rt.spec.y, 0),
                                        "x_shares": self.intended[key].get(rt.spec.x, 0)})
                else:
                    self.intended[key] = {}
                    changed = True
                self._log(now, rt, tag, bad_legs=",".join(sorted({rt.spec.y, rt.spec.x} & bad)))
                rt.pending = ""
            elif kind == "exit":
                tag = rt.sm.confirm(ok)
                if ok:
                    self._record_ledger(now, key, {})
                    for tr in reversed(self.trades):
                        if tr["trade_id"] == rt.trade_id:
                            tr["exit_fill"] = now
                            tr["exit_reason"] = rt.sm.exit_reason
                            tr["holding_sessions"] = rt.sm.held
                            break
                    rt.pending = ""
                self._log(now, rt, tag)
            elif kind == "rehedge":
                if ok:
                    self._record_ledger(now, key, self.intended[key])
                else:
                    self.intended[key][rt.spec.x] = rt.prev_x_shares
                    changed = True
                self._log(now, rt, "rehedge_confirmed" if ok else "rehedge_failed")
                rt.pending = ""

        # 2) per-pair decisions for active cohorts
        for plan in self.plans:
            c = plan.cohort
            if now < c.trade_start:
                continue
            ended = now >= c.trade_end
            expiring = nxt is not None and nxt >= c.trade_end and not ended
            if nxt is None:
                expiring = True
            for spec in plan.pairs:
                key = spec.key
                if key in self._done:
                    continue
                rt_opt = self.rt.get(key)
                if rt_opt is None:
                    if ended:
                        self._done.add(key)
                        continue
                    rt = self._activate(spec, c, hist)
                    self.rt[key] = rt
                    self.intended.setdefault(key, {})
                else:
                    rt = rt_opt
                if ended:
                    # window over: force flat (exit should already have filled)
                    if self.intended.get(key):
                        self.intended[key] = {}
                        rt.pending = "exit"
                        rt.sm.state = PairState.EXIT_PENDING
                        rt.sm.exit_reason = rt.sm.exit_reason or "window_expiry_forced"
                        changed = True
                    elif not rt.pending:
                        self._done.add(key)
                    continue
                if rt.pending:
                    continue
                z, beta_now, spread, missing = self._signal(rt, hist)
                bd = self._breakdown(rt, hist) if rt.sm.state is not PairState.SUSPENDED else False
                rt.sessions += 1
                if self.record_pairs is None or key in self.record_pairs:
                    self.series.append({"date": now, "key": key, "z": z, "beta": beta_now,
                                        "spread": spread, "state": rt.sm.state.value})
                was_open = rt.sm.is_open
                d = rt.sm.on_close(z, rt.last_z, breakdown=bd, expiring=expiring,
                                   missing=missing)
                rt.last_z = z
                if d.action in (Action.OPEN_LONG, Action.OPEN_SHORT):
                    if self._open(rt, d.action, beta_now, today, equity, now):
                        changed = True
                elif d.action is Action.CLOSE:
                    self.intended[key] = {}
                    rt.pending = "exit"
                    changed = True
                    self._log(now, rt, "exit_signal", reason=d.reason, z=z)
                elif (was_open and rt.sm.is_open and self.p.method != "static_ols"
                      and math.isfinite(beta_now) and self._maybe_rehedge(rt, beta_now, today,
                                                                          now)):
                    changed = True
                elif d.reason == "breakdown_suspend":
                    self._log(now, rt, "suspended_breakdown")

        if not changed:
            return None
        return self._emit(event, today, equity)

    def _open(self, rt: _PairRT, action: Action, beta: float, today: pd.Series,
              equity: float, now: pd.Timestamp) -> bool:
        sp, p = rt.spec, self.p
        key = sp.key

        def abort(reason: str) -> bool:
            rt.sm.state, rt.sm.pending_side = PairState.FLAT, None
            self._log(now, rt, "entry_skipped", reason=reason)
            return False

        live = [k for k, b in self.intended.items() if b and k != key]
        if any(self.rt[k].spec.pair == sp.pair for k in live if k in self.rt):
            return abort("duplicate_pair_live_in_other_cohort")
        for s in (sp.y, sp.x):
            n = sum(1 for k in live if s in self.intended[k])
            if n >= p.limits.max_pairs_per_name:
                return abort(f"max_pairs_per_name {s}")
        if not beta_allowed(beta, p.beta_min, p.beta_max):
            return abort(f"beta_policy {beta:.3f}")
        py, px = self._price(sp.y, today), self._price(sp.x, today)
        direction = 1 if action is Action.OPEN_LONG else -1
        gross = sp.gross_frac * equity
        lt = pair_holdings(beta, gross, direction, py, px, p.notional_rule)
        new_w = {sp.y: lt.y_notional / equity, sp.x: lt.x_notional / equity}
        k, why = admit_scale(self._current_weights(today, equity), new_w, self.sectors,
                             p.limits)
        if k <= 0:
            return abort(f"limits: {why}")
        if k < 1:
            lt = pair_holdings(beta, gross * k, direction, py, px, p.notional_rule)
        if lt.y_shares == 0 or lt.x_shares == 0:
            return abort("rounds_to_zero_shares")
        self.intended[key] = {sp.y: lt.y_shares, sp.x: lt.x_shares}
        rt.pending, rt.entry_beta, rt.direction = "entry", beta, direction
        self.holdings_audit.append({
            "date": now, "key": key, "pair": sp.pair, "direction": direction, "beta": beta,
            "rule": p.notional_rule, "scale": k, "price_y": py, "price_x": px,
            "y_notional_intended": lt.y_notional_intended,
            "x_notional_intended": lt.x_notional_intended, "y_shares": lt.y_shares,
            "x_shares": lt.x_shares, "y_notional": lt.y_notional, "x_notional": lt.x_notional,
            "gross": lt.gross, "net": lt.net, "dollar_imbalance": lt.dollar_imbalance,
            "rounding_gap": lt.rounding_gap})
        self._log(now, rt, "entry_signal", side=action.value, beta=beta, scale=k)
        return True

    def _maybe_rehedge(self, rt: _PairRT, beta: float, today: pd.Series,
                       now: pd.Timestamp) -> bool:
        if not math.isfinite(rt.entry_beta) or rt.entry_beta == 0:
            return False
        if abs(beta / rt.entry_beta - 1.0) <= self.p.rehedge_threshold:
            return False
        if not beta_allowed(beta, self.p.beta_min, self.p.beta_max):
            return False
        sp = rt.spec
        book = self.intended[sp.key]
        py, px = self._price(sp.y, today), self._price(sp.x, today)
        n_y = book.get(sp.y, 0) * py
        new_x = int(math.trunc(-beta * n_y / px))
        if new_x == book.get(sp.x, 0) or new_x == 0:
            return False
        rt.prev_x_shares = book.get(sp.x, 0)
        book[sp.x] = new_x
        rt.pending = "rehedge"
        self._log(now, rt, "rehedge_signal", beta_old=rt.entry_beta, beta_new=beta)
        rt.entry_beta = beta
        return True

    def _emit(self, event: MarketEvent, today: pd.Series, equity: float
              ) -> list[SignalEvent]:
        exp = self._expected()
        held = set(self._pf().positions)
        out = []
        for s in sorted(set(exp) | held):
            q = exp.get(s, 0)
            w = engine_weight_for_shares(q, self._price(s, today), equity) if q else 0.0
            out.append(self.signal(event, s, w))
        return out


def sectors_map(meta: pd.DataFrame) -> dict[str, str]:
    return dict(zip(meta["symbol"].astype(str), meta["sector"].astype(str), strict=True))
