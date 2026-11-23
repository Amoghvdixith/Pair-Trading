"""One selected pair end to end: prices, spread, z, OU fit, hedge paths, entries and fills.

    python scripts/run_single_pair.py [--data synthetic|market] [--mode fast|research]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pairs.experiments import run_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", choices=["synthetic", "market"], default="synthetic")
    ap.add_argument("--mode", choices=["fast", "research"], default="fast")
    a = ap.parse_args()
    run_all(Path(__file__).resolve().parents[1], a.data, a.mode, ("single_pair",))


if __name__ == "__main__":
    main()
