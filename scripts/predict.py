from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as `python scripts/predict.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.fp import predict as fp_predict
from src.gnn import predict as gnn_predict
from src.smiles import predict as smiles_predict
from src.common.uncertainty import resolve_model_artifact_dirs
from src.models import resolve_model_family


def _resolve_backend_from_model_dir(model_dir: Path) -> str:
    train_cfg_path = model_dir / "config_snapshot.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
    train_cfg = load_config(train_cfg_path)
    return resolve_model_family(train_cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict (dispatch to FP or GNN backend).")
    ap.add_argument("--config", required=True, help="Path to a composed predict config.")
    ap.add_argument("--query", required=True, help="CAS or formula depending on config.input.mode")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dirs = resolve_model_artifact_dirs(cfg)
    if not model_dirs:
        raise ValueError("model_artifact_dir(s) missing in config.")
    backend = _resolve_backend_from_model_dir(Path(model_dirs[0]))
    for model_dir in model_dirs[1:]:
        other_backend = _resolve_backend_from_model_dir(Path(model_dir))
        if other_backend != backend:
            raise ValueError(f"Ensemble backends do not match: {backend} vs {other_backend}")

    if backend == "fp":
        fp_predict.run(cfg, args.query)
    elif backend == "gnn":
        gnn_predict.run(cfg, args.query)
    elif backend == "smiles":
        smiles_predict.run(cfg, args.query)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()
