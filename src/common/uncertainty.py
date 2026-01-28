from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def normalize_uncertainty_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ucfg = cfg.get("uncertainty", {}) or {}
    method = str(ucfg.get("method", "")).strip().lower()
    if method in {"", "none", "off", "false", "0", "null"}:
        method = ""
    n_samples = ucfg.get("n_samples", ucfg.get("num_samples", 30))
    try:
        n_samples = int(n_samples)
    except Exception:
        n_samples = 30
    eps = ucfg.get("eps", 1.0e-6)
    try:
        eps = float(eps)
    except Exception:
        eps = 1.0e-6
    return {"method": method, "n_samples": n_samples, "eps": eps}


def _normalize_dirs(value: Any) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [Path(str(v)) for v in value if str(v).strip()]
    if isinstance(value, str) and not value.strip():
        return []
    return [Path(str(value))]


def resolve_model_artifact_dirs(cfg: Dict[str, Any]) -> List[Path]:
    dirs = _normalize_dirs(cfg.get("model_artifact_dirs"))
    if dirs:
        return dirs
    ensemble_cfg = cfg.get("ensemble", {}) or {}
    if isinstance(ensemble_cfg, dict):
        dirs = _normalize_dirs(ensemble_cfg.get("model_artifact_dirs"))
        if dirs:
            return dirs
    return _normalize_dirs(cfg.get("model_artifact_dir"))
