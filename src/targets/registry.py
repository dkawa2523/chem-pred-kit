from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TARGET_PREFIX = "target."


def target_column_name(name: str) -> str:
    name = str(name).strip()
    if name.startswith(TARGET_PREFIX):
        return name
    return f"{TARGET_PREFIX}{name}"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    aliases: List[str] = field(default_factory=list)
    units: Optional[str] = None
    transform: Optional[Dict[str, Any]] = None
    dtype: Optional[str] = None
    recommended_transform: Optional[Any] = None
    range_hint: Optional[Dict[str, Any]] = None
    missing_policy: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    source_type: Optional[str] = None
    quality_flag: Optional[Any] = None
    required: bool = True


class TargetRegistry:
    def __init__(self, targets: List[TargetSpec]) -> None:
        self._targets = targets
        self._by_name = {t.name: t for t in targets}
        alias_map: Dict[str, str] = {}
        for spec in targets:
            for alias in [spec.name, *spec.aliases]:
                alias = str(alias).strip()
                if not alias:
                    continue
                alias_map[alias] = spec.name
        self._alias_map = alias_map

    def canonical_names(self) -> List[str]:
        return [t.name for t in self._targets]

    def get(self, name: str) -> Optional[TargetSpec]:
        return self._by_name.get(name)

    def resolve_alias(self, name: str) -> Optional[str]:
        return self._alias_map.get(name)

    def resolve_target_columns(self, names: List[str]) -> List[str]:
        cols: List[str] = []
        for name in names:
            if name.startswith(TARGET_PREFIX):
                suffix = name[len(TARGET_PREFIX) :]
                canonical = self.resolve_alias(suffix) or suffix
                cols.append(target_column_name(canonical))
                continue
            canonical = self.resolve_alias(name) or name
            cols.append(target_column_name(canonical))
        return cols

    def resolve_target_names_from_columns(self, columns: List[str]) -> List[str]:
        names: List[str] = []
        for col in columns:
            name = col
            if col.startswith(TARGET_PREFIX):
                name = col[len(TARGET_PREFIX) :]
            name = self.resolve_alias(name) or name
            names.append(name)
        return names

    def resolve_source_column(self, target_name: str, available_columns: List[str]) -> Optional[str]:
        spec = self._by_name.get(target_name)
        if spec is None:
            return None
        for alias in [spec.name, *spec.aliases]:
            alias_str = str(alias).strip()
            if alias_str in available_columns:
                return alias_str
        return None

    def units_for(self, target_names: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name in target_names:
            spec = self._by_name.get(name)
            if spec and spec.units:
                out[name] = str(spec.units)
        return out

    def transform_config_for(self, target_name: str) -> Optional[Dict[str, Any]]:
        spec = self._by_name.get(target_name)
        if spec is None:
            return None
        return spec.transform

    def recommended_transform_for(self, target_name: str) -> Optional[Any]:
        spec = self._by_name.get(target_name)
        if spec is None:
            return None
        return spec.recommended_transform

    def metadata_for(self, target_name: str) -> Dict[str, Any]:
        spec = self._by_name.get(target_name)
        if spec is None:
            return {}
        return {
            "units": spec.units,
            "dtype": spec.dtype,
            "conditions": spec.conditions,
            "source_type": spec.source_type,
            "quality_flag": spec.quality_flag,
            "recommended_transform": spec.recommended_transform,
            "range_hint": spec.range_hint,
            "missing_policy": spec.missing_policy,
            "aliases": list(spec.aliases),
        }

    def target_metadata(self, target_names: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name in target_names:
            out[name] = self.metadata_for(name)
        return out


def _normalize_registry_cfg(cfg: Any) -> Optional[List[Dict[str, Any]]]:
    if cfg is None:
        return None
    if isinstance(cfg, list):
        return [item for item in cfg]
    if isinstance(cfg, dict):
        for key in ("targets", "registry", "items"):
            if key in cfg and isinstance(cfg[key], list):
                return cfg[key]
    return None


def load_target_registry(cfg: Optional[Dict[str, Any]]) -> Optional[TargetRegistry]:
    if not cfg:
        return None
    reg_cfg = _normalize_registry_cfg(cfg.get("target_registry"))
    if reg_cfg is None:
        return None
    targets: List[TargetSpec] = []
    for item in reg_cfg:
        if not isinstance(item, dict):
            raise ValueError("target_registry entries must be mappings")
        name = str(item.get("name") or item.get("canonical") or "").strip()
        if not name:
            raise ValueError("target_registry entries must include name")
        aliases = item.get("aliases") or []
        if isinstance(aliases, (tuple, list)):
            alias_list = [str(a) for a in aliases if str(a).strip()]
        else:
            alias_list = [str(aliases)]
        units = item.get("units")
        transform = item.get("transform")
        dtype = item.get("dtype")
        recommended_transform = item.get("recommended_transform")
        range_hint = item.get("range_hint")
        missing_policy = item.get("missing_policy")
        conditions = item.get("conditions") or item.get("condition")
        source_type = item.get("source_type")
        quality_flag = item.get("quality_flag")
        required = bool(item.get("required", True))
        targets.append(
            TargetSpec(
                name=name,
                aliases=alias_list,
                units=str(units) if units is not None else None,
                transform=transform if isinstance(transform, dict) else transform,
                dtype=str(dtype) if dtype is not None else None,
                recommended_transform=recommended_transform,
                range_hint=range_hint if isinstance(range_hint, dict) else range_hint,
                missing_policy=str(missing_policy) if missing_policy is not None else None,
                conditions=conditions if isinstance(conditions, dict) else conditions,
                source_type=str(source_type) if source_type is not None else None,
                quality_flag=quality_flag,
                required=required,
            )
        )
    if not targets:
        return None
    return TargetRegistry(targets)


def resolve_target_names(cfg: Dict[str, Any], target_columns: List[str]) -> List[str]:
    registry = load_target_registry(cfg)
    if registry is None:
        return [col[len(TARGET_PREFIX) :] if col.startswith(TARGET_PREFIX) else col for col in target_columns]
    return registry.resolve_target_names_from_columns(target_columns)


def resolve_target_units(cfg: Dict[str, Any], target_names: List[str]) -> Dict[str, str]:
    registry = load_target_registry(cfg)
    units = registry.units_for(target_names) if registry else {}
    dataset_units = cfg.get("dataset", {}).get("units", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
    for key, value in (dataset_units or {}).items():
        if value is None:
            continue
        units[str(key)] = str(value)
    return units


def resolve_target_metadata(cfg: Dict[str, Any], target_names: List[str]) -> Dict[str, Dict[str, Any]]:
    registry = load_target_registry(cfg)
    if registry is None:
        return {name: {} for name in target_names}
    return registry.target_metadata(target_names)
