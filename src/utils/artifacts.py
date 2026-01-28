from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from src.common.utils import save_json

REQUIRED_META_KEYS = [
    "run_id",
    "process_name",
    "created_at",
    "git_sha",
    "dataset_hash",
    "config_hash",
    "task_name",
    "model_name",
    "featureset_name",
    "upstream_artifacts",
    "tags",
]

REQUIRED_PREDICTION_COLUMNS = ["sample_id", "y_pred"]
REQUIRED_DATASET_META_KEYS = [
    "dataset_name",
    "dataset_hash",
    "target_names",
    "units",
    "dataset_csv",
    "indices_dir",
    "dataset_index",
    "split_json",
]
REQUIRED_DATASET_INDEX_COLUMNS = ["sample_id", "split"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_file_bytes(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def hash_files(paths: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        hasher.update(path.name.encode("utf-8"))
        for chunk in _iter_file_bytes(path):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataset_hash(dataset_csv: Path, indices_dir: Optional[Path] = None) -> Optional[str]:
    if dataset_csv is None:
        return None
    dataset_csv = Path(dataset_csv)
    if not dataset_csv.exists():
        return None
    paths = [dataset_csv]
    if indices_dir:
        indices_dir = Path(indices_dir)
        if indices_dir.exists():
            paths.extend(sorted([p for p in indices_dir.iterdir() if p.is_file()], key=lambda p: p.name))
    return hash_files(paths)


def compute_config_hash(cfg: Dict[str, Any]) -> Optional[str]:
    if cfg is None:
        return None
    payload = json.dumps(cfg, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return out.decode("utf-8").strip() or None


def resolve_task_name(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not cfg:
        return None
    task_cfg = cfg.get("task", {})
    if isinstance(task_cfg, dict) and task_cfg.get("name"):
        return str(task_cfg["name"])
    if cfg.get("task_name"):
        return str(cfg["task_name"])
    return None


def resolve_model_name(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not cfg:
        return None
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("name"):
        return str(model_cfg["name"])
    if cfg.get("model_name"):
        return str(cfg["model_name"])
    return None


def resolve_featureset_name(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not cfg:
        return None
    if cfg.get("featureset_name"):
        return str(cfg["featureset_name"])
    featureset_cfg = cfg.get("featureset", {})
    if isinstance(featureset_cfg, dict) and featureset_cfg.get("name"):
        return str(featureset_cfg["name"])
    features_cfg = cfg.get("features", {})
    if isinstance(features_cfg, dict) and features_cfg.get("name"):
        return str(features_cfg["name"])
    featurizer_cfg = cfg.get("featurizer", {})
    if isinstance(featurizer_cfg, dict) and featurizer_cfg.get("name"):
        return str(featurizer_cfg["name"])
    if isinstance(featurizer_cfg, dict) and featurizer_cfg.get("embedding_dim") is not None:
        return "pretrained_embedding"

    if isinstance(featurizer_cfg, dict):
        feat_name = str(featurizer_cfg.get("name", "")).lower()
        if feat_name in {"smiles", "smiles_tokenizer", "smiles-transformer", "smiles_transformer"}:
            return "smiles_tokenizer"
        if featurizer_cfg.get("tokenizer_name") or featurizer_cfg.get("pretrained_name"):
            return "smiles_tokenizer"

    if isinstance(featurizer_cfg, dict) and featurizer_cfg.get("fingerprint"):
        fp_name = str(featurizer_cfg.get("fingerprint", "fp"))
        name = f"fp_{fp_name}"
        if featurizer_cfg.get("add_descriptors"):
            name = f"{name}_desc"
        return name
    if isinstance(featurizer_cfg, dict) and (featurizer_cfg.get("node_features") or featurizer_cfg.get("edge_features")):
        return "graph"
    return None


def _resolve_dataset_paths(cfg: Dict[str, Any]) -> tuple[Optional[Path], Optional[Path]]:
    dataset_csv = None
    indices_dir = None
    data_cfg = cfg.get("data", {}) if cfg else {}
    if isinstance(data_cfg, dict):
        dataset_csv = data_cfg.get("dataset_csv") or dataset_csv
        indices_dir = data_cfg.get("indices_dir") or indices_dir
    paths_cfg = cfg.get("paths", {}) if cfg else {}
    if isinstance(paths_cfg, dict):
        dataset_csv = dataset_csv or paths_cfg.get("out_csv")
        indices_dir = indices_dir or paths_cfg.get("out_indices_dir")
    dataset_csv = dataset_csv or (cfg.get("dataset_csv") if cfg else None)
    indices_dir = indices_dir or (cfg.get("indices_dir") if cfg else None)
    return (Path(dataset_csv) if dataset_csv else None, Path(indices_dir) if indices_dir else None)


def compute_dataset_hash_from_cfg(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not cfg:
        return None
    dataset_csv, indices_dir = _resolve_dataset_paths(cfg)
    if dataset_csv is None:
        return None
    return compute_dataset_hash(dataset_csv, indices_dir)


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [str(t) for t in tags]
    if isinstance(tags, tuple):
        return [str(t) for t in tags]
    if isinstance(tags, str):
        return [tags]
    return [str(tags)]


def build_meta(
    process_name: str,
    cfg: Optional[Dict[str, Any]] = None,
    upstream_artifacts: Optional[Iterable[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    dataset_hash: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "run_id": uuid4().hex,
        "process_name": process_name,
        "created_at": utc_now_iso(),
        "upstream_artifacts": list(upstream_artifacts or []),
    }

    config_hash = compute_config_hash(cfg) if cfg is not None else None
    if dataset_hash is None and cfg is not None:
        dataset_hash = compute_dataset_hash_from_cfg(cfg)

    meta.update(
        {
            "git_sha": get_git_sha(),
            "dataset_hash": dataset_hash,
            "config_hash": config_hash,
            "task_name": resolve_task_name(cfg),
            "model_name": resolve_model_name(cfg),
            "featureset_name": resolve_featureset_name(cfg),
            "tags": _normalize_tags(cfg.get("tags") if cfg else None),
        }
    )

    if model_version is not None:
        meta["model_version"] = model_version

    if extra:
        meta.update(extra)

    for key in REQUIRED_META_KEYS:
        meta.setdefault(key, None)

    return meta


def save_meta(run_dir: str | Path, meta: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "meta.json"
    save_json(path, meta)
    return path


def load_meta(path_or_dir: str | Path) -> Dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "meta.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_training_context(
    train_cfg: Optional[Dict[str, Any]],
    train_meta: Optional[Dict[str, Any]] = None,
    model_artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    train_meta = train_meta or {}
    context = {
        "task_name": train_meta.get("task_name") or resolve_task_name(train_cfg),
        "model_name": train_meta.get("model_name") or resolve_model_name(train_cfg),
        "featureset_name": train_meta.get("featureset_name") or resolve_featureset_name(train_cfg),
        "featureset_hash": train_meta.get("featureset_hash"),
        "dataset_hash": train_meta.get("dataset_hash"),
        "model_version": train_meta.get("model_version") or train_meta.get("run_id"),
    }
    if context["model_version"] is None and model_artifact_dir is not None:
        context["model_version"] = model_artifact_dir.name
    return context


def validate_meta(meta: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"meta.json missing keys: {', '.join(missing)}")


def validate_predictions_csv(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"predictions.csv not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if not header:
        raise ValueError(f"predictions.csv missing header: {path}")
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"predictions.csv missing columns: {', '.join(missing)}")


def validate_dataset_index(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_index.csv not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if not header:
        raise ValueError(f"dataset_index.csv missing header: {path}")
    missing = [c for c in REQUIRED_DATASET_INDEX_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"dataset_index.csv missing columns: {', '.join(missing)}")


def validate_common_artifacts(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    meta_path = run_dir / "meta.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    validate_meta(load_meta(meta_path))


def validate_train_artifacts(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    validate_common_artifacts(run_dir)
    model_path = run_dir / "model" / "model.ckpt"
    metrics_path = run_dir / "metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model.ckpt not found: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")


def validate_evaluate_artifacts(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    validate_common_artifacts(run_dir)
    metrics_path = run_dir / "metrics.json"
    pred_path = run_dir / "predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    validate_predictions_csv(pred_path)


def validate_predict_artifacts(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    validate_common_artifacts(run_dir)
    pred_path = run_dir / "predictions.csv"
    validate_predictions_csv(pred_path)


def validate_dataset_artifacts(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    validate_common_artifacts(run_dir)
    meta = load_meta(run_dir)
    missing = [k for k in REQUIRED_DATASET_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"meta.json missing dataset keys: {', '.join(missing)}")

    dataset_csv = Path(meta["dataset_csv"])
    indices_dir = Path(meta["indices_dir"])
    dataset_index = Path(meta["dataset_index"])
    split_json = Path(meta["split_json"])

    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if not indices_dir.exists():
        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
    if not split_json.exists():
        raise FileNotFoundError(f"split.json not found: {split_json}")

    validate_dataset_index(dataset_index)

    for split_name in ["train", "val", "test"]:
        split_path = indices_dir / f"{split_name}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Split indices missing: {split_path}")
