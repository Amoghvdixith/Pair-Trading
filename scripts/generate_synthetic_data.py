"""Generate the labelled synthetic sector universe (offline, seeded).

    python scripts/generate_synthetic_data.py [--mode fast|research]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pairs.experiments import generate_synthetic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fast", "research"], default="fast")
    a = ap.parse_args()
    print(generate_synthetic(Path(__file__).resolve().parents[1], a.mode))


if __name__ == "__main__":
    main()
