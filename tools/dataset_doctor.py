from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow running as `python -m tools.dataset_doctor ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config, load_yaml
from src.common.utils import get_logger
from src.data.adapters.registry import create_dataset_adapter
from src.data.adapters.utils import resolve_dataset_name
from src.utils.artifacts import validate_dataset_artifacts


def _find_dataset_config(dataset_name: str, root: Path) -> Optional[Path]:
    configs_dir = root / "configs" / "dataset"
    if not configs_dir.exists():
        return None
    for path in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = load_yaml(path)
        except Exception:
            continue
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
        name = dataset_cfg.get("name") if isinstance(dataset_cfg, dict) else None
        if name == dataset_name:
            return path
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate or build dataset artifacts via adapter registry.")
    ap.add_argument("--dataset", default=None, help="Dataset name (matches dataset.name in configs/dataset/*.yaml).")
    ap.add_argument("--config", default=None, help="Path to dataset config (overrides --dataset lookup).")
    ap.add_argument("--build", action="store_true", help="Force rebuild the dataset artifacts.")
    args = ap.parse_args()

    if args.config:
        cfg_path = Path(args.config)
        cfg = load_config(cfg_path)
    elif args.dataset:
        cfg_path = _find_dataset_config(args.dataset, REPO_ROOT)
        if cfg_path is None:
            raise FileNotFoundError(f"No dataset config found for dataset.name={args.dataset}")
        cfg = load_config(cfg_path)
    else:
        raise ValueError("Provide --dataset or --config.")

    cfg.setdefault("process", {"name": "build_dataset"})
    dataset_name = resolve_dataset_name(cfg, default=args.dataset)
    if not dataset_name:
        raise ValueError("dataset.name missing and no --dataset provided.")

    paths = cfg.get("paths", {})
    out_csv = Path(paths.get("out_csv", "data/processed/dataset_with_lj.csv"))
    run_dir = Path(cfg.get("output", {}).get("run_dir", out_csv.parent))
    logger = get_logger("dataset_doctor", log_file=run_dir / "dataset_doctor.log")

    need_build = args.build
    if not need_build:
        indices_dir = Path(paths.get("out_indices_dir", "data/processed/indices"))
        if not out_csv.exists() or not indices_dir.exists():
            need_build = True

    if need_build:
        adapter = create_dataset_adapter(str(dataset_name))
        logger.info("Building dataset '%s' using adapter '%s'", dataset_name, adapter.name)
        adapter.build_processed(cfg, logger)

    validate_dataset_artifacts(run_dir)
    logger.info("Dataset artifacts OK: %s", run_dir)


if __name__ == "__main__":
    main()
