"""Run the full research pipeline (offline synthetic by default).

    python scripts/run_all.py                    # synthetic, fast mode
    python scripts/run_all.py --mode research    # larger universe, monthly cohorts
    python scripts/run_all.py --data market      # requires scripts/download_market_data.py
    python scripts/run_all.py --steps screening portfolio

The locked hold-out is NEVER evaluated here (see scripts/run_holdout.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pairs.experiments import run_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["synthetic", "market"], default="synthetic")
    ap.add_argument("--mode", choices=["fast", "research"], default="fast")
    ap.add_argument("--steps", nargs="*", default=None,
                    choices=["examples", "calibration", "screening", "single_pair", "portfolio",
                             "hedge", "costs", "thresholds", "stability"])
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_all(root, a.data, a.mode, tuple(a.steps) if a.steps else None)


if __name__ == "__main__":
    main()
