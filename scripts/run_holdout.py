"""Evaluate the frozen configuration on the LOCKED hold-out (never run by run_all.py).

    python scripts/run_holdout.py --confirm

Each evaluation is logged in results/logs/holdout_evaluations.jsonl.  After you look at
the hold-out, it is no longer untouched: do not retune and re-evaluate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pairs.experiments import holdout_evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", choices=["synthetic", "market"], default="synthetic")
    ap.add_argument("--confirm", action="store_true", help="required")
    a = ap.parse_args()
    out = holdout_evaluation(Path(__file__).resolve().parents[1], a.data, "fast", a.confirm)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
