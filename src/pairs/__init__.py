"""pairs - an educational pairs-trading (statistical arbitrage) research pipeline.

Research software, not a trading system.  Statistical arbitrage is NOT risk-free
arbitrage: cointegration can break, spreads can widen further, and costs, borrow and
financing can turn a statistically stationary spread into a losing strategy.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .hedge import KalmanHedge, kalman_filter, ols_hedge, rolling_ols_hedge
from .ou import OUFit, fit_ou, hurst_exponent, variance_ratio_test
from .screening import (
    benjamini_hochberg,
    benjamini_yekutieli,
    engle_granger,
    johansen,
    screen_pairs,
)
from .signals import PairState, PairStateMachine, SignalConfig, rolling_zscore

__all__ = [
    "__version__", "KalmanHedge", "OUFit", "PairState", "PairStateMachine", "SignalConfig",
    "benjamini_hochberg", "benjamini_yekutieli", "engle_granger", "fit_ou", "hurst_exponent",
    "johansen", "kalman_filter", "ols_hedge", "rolling_ols_hedge", "rolling_zscore",
    "screen_pairs", "variance_ratio_test",
]
