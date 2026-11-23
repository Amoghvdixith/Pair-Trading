"""Command-line entry point: ``pairs <command>`` (installed console script)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import generate_synthetic, holdout_evaluation, project_root, run_all

STEPS = {"screen": ("screening", "stability"), "hedge": ("hedge",),
         "single-pair": ("single_pair",), "portfolio": ("portfolio",),
         "sensitivity": ("costs", "thresholds"),
         "calibration": ("examples", "calibration")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pairs", description="Pairs-trading research pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-data", help="write the synthetic dataset")
    g.add_argument("--mode", choices=["fast", "research"], default="fast")
    for name in (*STEPS, "run-all"):
        p = sub.add_parser(name)
        p.add_argument("--data", choices=["synthetic", "market"], default="synthetic")
        p.add_argument("--mode", choices=["fast", "research"], default="fast")
    h = sub.add_parser("holdout", help="evaluate the locked hold-out (requires --confirm)")
    h.add_argument("--data", choices=["synthetic", "market"], default="synthetic")
    h.add_argument("--confirm", action="store_true")
    a = ap.parse_args(argv)
    root = project_root(Path.cwd())
    if a.cmd == "generate-data":
        print(generate_synthetic(root, a.mode))
    elif a.cmd == "holdout":
        print(json.dumps(holdout_evaluation(root, a.data, "fast", a.confirm), default=str,
                         indent=2))
    else:
        run_all(root, a.data, a.mode, None if a.cmd == "run-all" else STEPS[a.cmd])


if __name__ == "__main__":
    main()
