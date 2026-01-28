from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ConfigError(f"Empty config: {path}")
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a mapping/dict: {path}")
    return cfg


def _deep_merge(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _iter_defaults(defaults: Iterable[Any]) -> Iterable[tuple[str, str] | str]:
    for entry in defaults:
        if isinstance(entry, str):
            yield entry
            continue
        if isinstance(entry, dict) and len(entry) == 1:
            group, name = next(iter(entry.items()))
            if not isinstance(group, str) or not isinstance(name, str):
                raise ConfigError(f"Invalid defaults entry: {entry!r}")
            yield (group, name)
            continue
        raise ConfigError(f"Invalid defaults entry: {entry!r}")


def _find_group_path(base_dir: Path, group: str, name: str) -> Path:
    current = base_dir
    group_path = Path(group)
    while True:
        candidates = [
            current / group_path / f"{name}.yaml",
            current / "configs" / group_path / f"{name}.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if current.parent == current:
            break
        current = current.parent
    raise ConfigError(f"Config file not found for defaults entry: {group}/{name}")


def _compose_from_cfg(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    defaults = cfg.get("defaults")
    if defaults is None:
        return cfg
    if not isinstance(defaults, list):
        raise ConfigError("defaults must be a list")

    composed: Dict[str, Any] = {}
    merge_self = False
    for entry in _iter_defaults(defaults):
        if entry == "_self_":
            merge_self = True
            continue
        if isinstance(entry, str):
            if "/" not in entry:
                raise ConfigError(f"Defaults entry must be group/name or dict: {entry!r}")
            group, name = entry.split("/", 1)
        else:
            group, name = entry

        group_path = _find_group_path(base_dir, group, name)
        group_cfg = load_yaml(group_path)
        group_cfg = _compose_from_cfg(group_cfg, group_path.parent)
        composed = _deep_merge(composed, group_cfg)

    if merge_self:
        self_cfg = {k: v for k, v in cfg.items() if k != "defaults"}
        composed = _deep_merge(composed, self_cfg)

    return composed


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)
    return _compose_from_cfg(cfg, path.parent)


def dump_yaml(path: str | Path, cfg: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
