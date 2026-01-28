from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/benchmark_suite.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.benchmark_suite import run
from src.common.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark suite matrix (train/evaluate/visualize/leaderboard).")
    ap.add_argument("--config", required=True, help="Path to a composed benchmark suite config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
