from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/leaderboard.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.common.leaderboard import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate runs into a leaderboard.")
    ap.add_argument("--config", required=True, help="Path to a composed leaderboard config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
