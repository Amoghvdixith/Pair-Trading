"""Install the project-02 backtester from a local folder WITHOUT writing into that folder.

The source tree is copied to a temporary directory, a wheel is built there and installed
into the current interpreter.  No Git operation is used.

    python scripts/install_local_backtester.py              # default: ../backtester
    python scripts/install_local_backtester.py --path D:/code/backtester
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def source_hash(src: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((src / "src" / "backtester").rglob("*.py")):
        h.update(p.relative_to(src).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--path", default=str(root.parent / "backtester"),
                    help="folder containing the backtester's pyproject.toml")
    args = ap.parse_args()
    src = Path(args.path).resolve()
    if not (src / "pyproject.toml").exists() or not (src / "src" / "backtester").is_dir():
        print(f"backtester not found at {src}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "backtester_copy"
        t.mkdir()
        shutil.copy2(src / "pyproject.toml", t)
        if (src / "README.md").exists():
            shutil.copy2(src / "README.md", t)
        shutil.copytree(src / "src", t / "src",
                        ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"))
        wheels = Path(tmp) / "wheels"
        subprocess.check_call([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q",
                               "-w", str(wheels), str(t)])
        whl = next(wheels.glob("backtester-*.whl"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                               "--force-reinstall", str(whl)])
    print(f"installed {whl.name}; source sha256 {source_hash(src)[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
