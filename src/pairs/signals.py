"""Causal z-scores and the per-pair signal state machine.

z-score (static-OLS method)
---------------------------
z_t = (s_t - mean(s_{t-L+1..t})) / sd(s_{t-L+1..t})  with ``include_current=True``
(default: the current COMPLETED close is part of the normalisation, which is causal
because the signal is acted on at the next open).  ``include_current=False`` uses
s_{t-L..t-1}.  A window with sd below ``min_sd`` yields NaN (no signal), never inf.

State machine
-------------
States: FLAT, ENTRY_PENDING, LONG_SPREAD, SHORT_SPREAD, EXIT_PENDING, SUSPENDED.
Orders created at close t fill no earlier than the next open; the pair is marked
LONG/SHORT only after BOTH legs are confirmed filled at the following close
(:meth:`PairStateMachine.confirm`).  A failed leg triggers the unwind policy.

Precedence at each close while a trade is open (first match wins):
  1. trading-window expiry      -> close
  2. relationship breakdown     -> close, then SUSPENDED for the rest of the window
  3. missing data               -> hold; close after ``max_missing`` consecutive misses
  4. undefined z (NaN)          -> hold
  5. stop loss |z| >= stop_z    -> close, then SUSPENDED
  6. time stop (held >= limit)  -> close (cool-down applies)
  7. take profit (exit band)    -> close (cool-down applies)
While FLAT: expiry/missing/NaN -> nothing; breakdown -> SUSPENDED; |z| >= stop_z blocks
entry; cool-down blocks entry; otherwise entry by ``exceed`` (|z| >= entry_z) or ``cross``
(threshold crossed between t-1 and t).
Daily-bar stops are evaluated at the close and executed at the next open: the fill can
be far beyond the threshold (gap risk).  Nothing guarantees a fill at the stop level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series, window: int, include_current: bool = True,
                   min_periods: int | None = None, ddof: int = 1, min_sd: float = 1e-10
                   ) -> pd.Series:
    """Causal rolling z-score (see module docstring)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    s = pd.Series(spread, dtype=float)
    mp = window if min_periods is None else min_periods
    base = s if include_current else s.shift(1)
    r = base.rolling(window, min_periods=mp)
    mean, sd = r.mean(), r.std(ddof=ddof)
    return (s - mean) / sd.where(sd > min_sd)


def max_attainable_z(window: int, include_current: bool = True) -> float:
    """Largest possible |z| of the last point when it is part of its own window.

    With the current value included in a sample of size L (ddof=1), Samuelson's
    inequality gives |z| <= (L - 1) / sqrt(L).  Thresholds above this can NEVER trigger.
    Excluding the current value removes the bound (returns inf).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    return (window - 1) / math.sqrt(window) if include_current else math.inf


def zscore_last(values: np.ndarray, min_sd: float = 1e-10) -> float:
    """z of the last element against the whole array (include-current convention)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or not np.all(np.isfinite(v)):
        return math.nan
    sd = float(np.std(v, ddof=1))
    if sd <= min_sd:
        return math.nan
    return float((v[-1] - v.mean()) / sd)


class PairState(StrEnum):
    FLAT = "flat"
    ENTRY_PENDING = "entry_pending"
    LONG_SPREAD = "long_spread"    # long y (A), short beta-weighted x (B)
    SHORT_SPREAD = "short_spread"  # short y, long x
    EXIT_PENDING = "exit_pending"
    SUSPENDED = "suspended"


class Action(StrEnum):
    NONE = "none"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"


@dataclass(frozen=True)
class SignalConfig:
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.5
    entry_mode: str = "exceed"
    time_stop: int = 30
    cooldown: int = 5
    max_missing: int = 3

    def __post_init__(self) -> None:
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("need 0 <= exit_z < entry_z < stop_z")
        if self.entry_mode not in ("exceed", "cross"):
            raise ValueError("entry_mode must be exceed or cross")
        if self.time_stop < 1 or self.cooldown < 0 or self.max_missing < 0:
            raise ValueError("time_stop >= 1, cooldown >= 0, max_missing >= 0")


@dataclass
class Decision:
    action: Action
    reason: str = ""


@dataclass
class PairStateMachine:
    cfg: SignalConfig
    state: PairState = PairState.FLAT
    held: int = 0
    cooldown_left: int = 0
    missing_run: int = 0
    pending_side: PairState | None = None
    suspend_after_exit: bool = False
    exit_reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ fills
    def confirm(self, filled: bool) -> str:
        """Called at the close after an order's eligible open.  Returns a log tag."""
        if self.state is PairState.ENTRY_PENDING:
            if filled:
                assert self.pending_side is not None
                self.state, self.held = self.pending_side, 0
                return "entry_confirmed"
            self.state, self.cooldown_left = PairState.FLAT, self.cfg.cooldown
            self.pending_side = None
            return "leg_failure"
        if self.state is PairState.EXIT_PENDING:
            if not filled:
                return "exit_incomplete"  # adapter keeps re-sending the flat target
            self.state = PairState.SUSPENDED if self.suspend_after_exit else PairState.FLAT
            self.cooldown_left = self.cfg.cooldown
            self.pending_side = None
            return "exit_confirmed"
        return ""

    @property
    def is_open(self) -> bool:
        return self.state in (PairState.LONG_SPREAD, PairState.SHORT_SPREAD)

    # ------------------------------------------------------------------ decisions
    def on_close(self, z: float, z_prev: float = math.nan, *, breakdown: bool = False,
                 expiring: bool = False, missing: bool = False) -> Decision:
        d = self._decide(z, z_prev, breakdown, expiring, missing)
        if d.action is Action.CLOSE:
            self.state = PairState.EXIT_PENDING
            self.exit_reason = d.reason
            self.suspend_after_exit = d.reason in ("breakdown", "stop_loss")
        elif d.action in (Action.OPEN_LONG, Action.OPEN_SHORT):
            self.state = PairState.ENTRY_PENDING
            self.pending_side = (PairState.LONG_SPREAD if d.action is Action.OPEN_LONG
                                 else PairState.SHORT_SPREAD)
        return d

    def _decide(self, z: float, z_prev: float, breakdown: bool, expiring: bool,
                missing: bool) -> Decision:
        c = self.cfg
        if self.state in (PairState.ENTRY_PENDING, PairState.EXIT_PENDING,
                          PairState.SUSPENDED):
            return Decision(Action.NONE, self.state.value)
        if self.is_open:
            self.held += 1
            if expiring:
                return Decision(Action.CLOSE, "window_expiry")
            if breakdown:
                return Decision(Action.CLOSE, "breakdown")
            if missing:
                self.missing_run += 1
                if self.missing_run > c.max_missing:
                    return Decision(Action.CLOSE, "missing_data")
                return Decision(Action.NONE, "missing_hold")
            self.missing_run = 0
            if not math.isfinite(z):
                return Decision(Action.NONE, "z_undefined_hold")
            if abs(z) >= c.stop_z:
                return Decision(Action.CLOSE, "stop_loss")
            if self.held >= c.time_stop:
                return Decision(Action.CLOSE, "time_stop")
            if self.state is PairState.LONG_SPREAD and z >= -c.exit_z:
                return Decision(Action.CLOSE, "take_profit")
            if self.state is PairState.SHORT_SPREAD and z <= c.exit_z:
                return Decision(Action.CLOSE, "take_profit")
            return Decision(Action.NONE, "hold")
        # FLAT
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        if breakdown:
            self.state = PairState.SUSPENDED
            return Decision(Action.NONE, "breakdown_suspend")
        if expiring or missing or not math.isfinite(z):
            return Decision(Action.NONE, "no_entry")
        if abs(z) >= c.stop_z:
            return Decision(Action.NONE, "stop_range_blocks_entry")
        if self.cooldown_left > 0:
            return Decision(Action.NONE, "cooldown")
        if c.entry_mode == "exceed":
            if z >= c.entry_z:
                return Decision(Action.OPEN_SHORT, "entry")
            if z <= -c.entry_z:
                return Decision(Action.OPEN_LONG, "entry")
        else:
            if math.isfinite(z_prev):
                if z_prev < c.entry_z <= z:
                    return Decision(Action.OPEN_SHORT, "entry_cross")
                if z_prev > -c.entry_z >= z:
                    return Decision(Action.OPEN_LONG, "entry_cross")
        return Decision(Action.NONE, "flat")


def run_state_machine(z: pd.Series, cfg: SignalConfig,
                      breakdown: pd.Series | None = None) -> pd.DataFrame:
    """Offline helper: replay a z path assuming every order fills at the next bar.

    Used for hand-checkable tests and diagnostics (the backtest adapter drives the
    machine itself with real fill confirmations).
    """
    sm = PairStateMachine(cfg)
    rows = []
    prev = math.nan
    pending = False
    for t, zt in z.items():
        if pending:
            sm.confirm(True)
            pending = False
        bd = bool(breakdown.loc[t]) if breakdown is not None else False
        d = sm.on_close(float(zt), prev, breakdown=bd)
        if d.action is not Action.NONE:
            pending = True
        rows.append({"date": t, "z": zt, "action": d.action.value, "reason": d.reason,
                     "state_after": sm.state.value})
        prev = float(zt)
    return pd.DataFrame(rows).set_index("date")
