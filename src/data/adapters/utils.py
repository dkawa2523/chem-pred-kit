from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_dataset_name(cfg: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
    dataset_cfg = cfg.get("dataset", {}) if cfg else {}
    if isinstance(dataset_cfg, dict) and dataset_cfg.get("name"):
        return str(dataset_cfg.get("name"))
    if cfg and cfg.get("dataset_name"):
        return str(cfg.get("dataset_name"))
    return default
