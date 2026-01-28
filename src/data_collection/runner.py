from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from src.common.config import dump_yaml
from src.common.io import write_csv
from src.common.meta import build_meta, save_meta
from src.common.utils import ensure_dir, get_logger
from src.data_collection.cache import FileCache
from src.data_collection.registry import create_data_source
from src.data_collection.types import DataCollectionQuery
from src.utils.artifacts import hash_files
from src.utils.validate_config import validate_config


def _parse_query(source_cfg: Dict[str, Any], collection_cfg: Dict[str, Any]) -> DataCollectionQuery:
    query_cfg = dict(source_cfg.get("query", {}) or {})
    query_cfg.update(collection_cfg.get("query", {}) or {})
    identifiers = query_cfg.get("identifiers", []) or []
    filters = query_cfg.get("filters", {}) or {}
    limit = query_cfg.get("limit")
    return DataCollectionQuery(identifiers=[str(x) for x in identifiers], filters=dict(filters), limit=limit)


def _write_sdf_records(sdf_dir: Path, sdf_records: Dict[str, str], overwrite: bool) -> list[Path]:
    paths = []
    if not sdf_records:
        return paths
    sdf_dir.mkdir(parents=True, exist_ok=True)
    for key, sdf_text in sdf_records.items():
        path = sdf_dir / f"{key}.sdf"
        if path.exists() and not overwrite:
            continue
        path.write_text(str(sdf_text), encoding="utf-8")
        paths.append(path)
    return paths


def _maybe_export(path: Path, target: Optional[str | Path], overwrite: bool) -> Optional[Path]:
    if not target:
        return None
    target_path = Path(target)
    if target_path.exists() and not overwrite:
        return None
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target_path)
    return target_path


def _maybe_export_dir(src_dir: Path, target_dir: Optional[str | Path], overwrite: bool) -> Optional[Path]:
    if not target_dir:
        return None
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.glob("*.sdf")):
        dest = target_dir / path.name
        if dest.exists() and not overwrite:
            continue
        shutil.copy2(path, dest)
    return target_dir


def _compute_raw_hash(raw_csv: Path, sdf_paths: Iterable[Path]) -> Optional[str]:
    paths = [raw_csv]
    for path in sdf_paths:
        if path.exists():
            paths.append(path)
    return hash_files(paths) if paths else None


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)

    output_cfg = cfg.get("output", {}) or {}
    experiment_cfg = cfg.get("experiment", {}) or {}
    exp_name = str(output_cfg.get("exp_name", experiment_cfg.get("name", "collect_data")))
    run_dir = ensure_dir(Path(output_cfg.get("run_dir", "runs/collect_data")) / exp_name)
    raw_dir = ensure_dir(run_dir / "raw")

    logger = get_logger("collect_data", log_file=run_dir / "collect_data.log")

    dump_yaml(run_dir / "config.yaml", cfg)

    collection_cfg = cfg.get("data_collection", {}) or {}
    source_cfg = cfg.get("data_source", {}) or {}

    source_name = str(source_cfg.get("name", "")).strip()
    if not source_name:
        raise ValueError("data_source.name is required")

    cache_cfg = collection_cfg.get("cache", {}) or {}
    cache = None
    if bool(cache_cfg.get("enabled", False)):
        cache_dir = cache_cfg.get("dir", ".cache/data_collection")
        cache = FileCache(cache_dir)

    query = _parse_query(source_cfg, collection_cfg)
    data_source = create_data_source(source_name, source_cfg, collection_cfg, cache)

    formatter_cfg = source_cfg.get("formatter", {}) or {}
    formatter_cfg.setdefault("sample_id_column", collection_cfg.get("sample_id_column", "sample_id"))
    result = data_source.collect(query, formatter_cfg, logger)

    required_columns = collection_cfg.get("required_columns", []) or []
    if required_columns:
        missing = [col for col in required_columns if col not in result.table.columns]
        if missing:
            logger.warning("Missing required columns in collected table: %s", ", ".join(missing))

    output_spec = collection_cfg.get("output", {}) or {}
    raw_csv_name = str(output_spec.get("raw_csv_name", "raw.csv"))
    sdf_dir_name = str(output_spec.get("sdf_dir_name", "sdf_files"))

    raw_csv_path = raw_dir / raw_csv_name
    write_csv(result.table, raw_csv_path)

    sdf_dir = raw_dir / sdf_dir_name
    sdf_paths = _write_sdf_records(sdf_dir, result.sdf_records, overwrite=bool(output_spec.get("overwrite", False)))

    export_cfg = collection_cfg.get("export", {}) or {}
    export_enabled = bool(export_cfg.get("enabled", False))
    overwrite_export = bool(export_cfg.get("overwrite", False))
    if export_enabled:
        _maybe_export(raw_csv_path, export_cfg.get("raw_csv"), overwrite_export)
        if sdf_paths:
            _maybe_export_dir(sdf_dir, export_cfg.get("sdf_dir"), overwrite_export)

    dataset_hash = _compute_raw_hash(raw_csv_path, sdf_paths)
    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "collect_data")),
        cfg=cfg,
        dataset_hash=dataset_hash,
        extra={"data_source": source_name},
    )
    save_meta(run_dir, meta)

    logger.info("Collected %d rows.", len(result.table))
    return run_dir
