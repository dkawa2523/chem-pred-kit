from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as `python scripts/pretrain.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.gnn import pretrain_graphcl as gnn_pretrain
from src.models import resolve_model_family


def _resolve_backend(cfg: Dict[str, Any]) -> str:
    process_cfg = cfg.get("process", {})
    backend = str(process_cfg.get("backend", "")).lower()
    if backend:
        return backend
    return resolve_model_family(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-supervised pretraining (GraphCL).")
    ap.add_argument("--config", required=True, help="Path to a composed pretrain config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = _resolve_backend(cfg)
    if backend == "gnn":
        gnn_pretrain.run(cfg)
    else:
        raise ValueError(f"Unsupported backend for pretrain: {backend}")


if __name__ == "__main__":
    main()
