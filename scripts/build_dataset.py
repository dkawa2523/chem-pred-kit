from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as `python scripts/build_dataset.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.common.utils import ensure_dir, get_logger, save_json
from src.data.adapters.registry import create_dataset_adapter
from src.data.adapters.utils import resolve_dataset_name
from src.data.audit import audit_dataset
from src.utils.validate_config import validate_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Build processed dataset via dataset adapter registry.")
    ap.add_argument("--config", required=True, help="Path to configs/dataset.yaml")
    ap.add_argument("--limit", type=int, default=None, help="Debug: limit number of rows loaded from raw CSV.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["limit_rows"] = int(args.limit)

    validate_config(cfg)
    paths = cfg.get("paths", {})
    out_csv = Path(paths.get("out_csv", "data/processed/dataset_with_lj.csv"))
    logger = get_logger("build_dataset", log_file=out_csv.parent / "build_dataset.log")

    dataset_name = resolve_dataset_name(cfg, default="lj")
    if not cfg.get("dataset", {}).get("name"):
        logger.warning("dataset.name not set; defaulting to '%s'.", dataset_name)

    adapter = create_dataset_adapter(str(dataset_name))
    logger.info("Using dataset adapter: %s", dataset_name)
    artifacts = adapter.build_processed(cfg, logger)

    audit_cfg = cfg.get("audit", {}) or {}
    enabled = audit_cfg.get("enabled", audit_cfg.get("enable", "auto"))
    enabled_str = str(enabled).lower() if isinstance(enabled, str) else enabled
    should_run = enabled_str in {"auto", "true", "1", "yes"} or enabled is True
    if should_run:
        try:
            report, report_md, _ = audit_dataset(cfg)
        except ImportError as exc:
            if enabled is True or str(enabled).lower() == "true":
                raise
            logger.warning("Skipping dataset audit (missing dependency): %s", exc)
        except Exception as exc:
            logger.warning("Dataset audit failed: %s", exc)
        else:
            out_dir = Path(audit_cfg.get("output_dir", artifacts.dataset_csv.parent))
            ensure_dir(out_dir)
            report_path = out_dir / "label_report.json"
            md_path = out_dir / "label_report.md"
            save_json(report_path, report)
            md_path.write_text(report_md, encoding="utf-8")
            logger.info("Saved label report to %s", report_path)
            meta_path = artifacts.meta_path
            if meta_path and Path(meta_path).exists():
                try:
                    import json

                    with Path(meta_path).open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["label_report"] = str(report_path)
                    meta["label_report_md"] = str(md_path)
                    save_json(Path(meta_path), meta)
                except Exception as exc:
                    logger.warning("Failed to update meta with label_report: %s", exc)


if __name__ == "__main__":
    main()
