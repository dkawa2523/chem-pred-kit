from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set, Tuple

from src.featuresets.base import FeatureSetBase, GraphFeatureSet, SmilesFeatureSet, TabularFeatureSet
from src.models.heads import apply_head_override

ModelFactory = Callable[[Dict[str, Any], Dict[str, Any]], Any]

_ALLOWED_INPUTS = {"pos", "edge_attr", "angles", "tokens"}
_MODEL_REGISTRY: Dict[str, "ModelSpec"] = {}
_MODEL_FAMILIES: Dict[str, Set[str]] = {}


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True)
class ModelCapabilities:
    supports_3d: bool = False
    supports_multitask: bool = False
    supports_force: bool = False
    requires_3d_pos: bool = False
    requires_distance_edges: bool = False
    accepts_global_descriptors: bool = True
    input_requires: Tuple[str, ...] = ()

    def normalized_inputs(self) -> Set[str]:
        return {_normalize_name(v) for v in self.input_requires if str(v).strip()}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    factory: ModelFactory
    capabilities: ModelCapabilities


def register_model(
    name: str,
    family: str,
    factory: ModelFactory,
    capabilities: Optional[ModelCapabilities] = None,
) -> None:
    name_key = _normalize_name(name)
    family_key = _normalize_name(family)
    if not name_key:
        raise ValueError("Model name must be non-empty.")
    if not family_key:
        raise ValueError("Model family must be non-empty.")
    if capabilities is None:
        capabilities = ModelCapabilities()
    normalized_inputs = capabilities.normalized_inputs()
    unknown = sorted(normalized_inputs - _ALLOWED_INPUTS)
    if unknown:
        raise ValueError(f"Unknown input_requires for model '{name_key}': {', '.join(unknown)}")
    capabilities = replace(capabilities, input_requires=tuple(sorted(normalized_inputs)))
    if name_key in _MODEL_REGISTRY:
        raise ValueError(f"Model already registered: {name_key}")
    spec = ModelSpec(name=name_key, family=family_key, factory=factory, capabilities=capabilities)
    _MODEL_REGISTRY[name_key] = spec
    _MODEL_FAMILIES.setdefault(family_key, set()).add(name_key)


def _ensure_registry_loaded() -> None:
    if _MODEL_REGISTRY:
        return
    import importlib

    importlib.import_module("src.models.builtin")


def list_models(family: Optional[str] = None) -> Sequence[str]:
    _ensure_registry_loaded()
    if family:
        family_key = _normalize_name(family)
        return sorted(_MODEL_FAMILIES.get(family_key, set()))
    return sorted(_MODEL_REGISTRY.keys())


def get_model_spec(name: str) -> ModelSpec:
    _ensure_registry_loaded()
    name_key = _normalize_name(name)
    spec = _MODEL_REGISTRY.get(name_key)
    if spec is None:
        available = ", ".join(list_models()) or "<none>"
        raise ValueError(f"Unknown model.name: {name}. Available: {available}")
    return spec


def resolve_model_name(cfg: Dict[str, Any], default: Optional[str] = None) -> str:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    name = None
    if isinstance(model_cfg, dict):
        name = model_cfg.get("name")
    if not name:
        name = default
    if not name:
        raise ValueError("model.name is required in config.")
    return _normalize_name(name)


def resolve_model_spec(cfg: Dict[str, Any], default: Optional[str] = None) -> ModelSpec:
    name = resolve_model_name(cfg, default=default)
    return get_model_spec(name)


def resolve_model_family(cfg: Dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if isinstance(model_cfg, dict) and model_cfg.get("family"):
        return _normalize_name(model_cfg.get("family"))
    name = resolve_model_name(cfg)
    return get_model_spec(name).family


def create_model(
    name: str,
    model_cfg: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    spec = get_model_spec(name)
    model = spec.factory(model_cfg=model_cfg or {}, context=context or {})
    return apply_head_override(model, model_cfg or {}, context=context or {})


def feature_inputs_from_featureset(featureset: FeatureSetBase) -> Set[str]:
    inputs: Set[str] = set()
    if isinstance(featureset, GraphFeatureSet):
        gcfg = featureset.pipeline.graph_cfg
        edge_features = list(getattr(gcfg, "edge_features", []) or [])
        if edge_features:
            inputs.add("edge_attr")
        if bool(getattr(gcfg, "use_3d_pos", False)):
            inputs.add("pos")
        angle_features = list(getattr(gcfg, "angle_features", []) or [])
        if angle_features:
            inputs.add("angles")
    elif isinstance(featureset, TabularFeatureSet):
        pass
    elif isinstance(featureset, SmilesFeatureSet):
        inputs.add("tokens")
    return inputs


def validate_model_requirements(
    spec: ModelSpec,
    target_columns: Sequence[str],
    feature_inputs: Iterable[str],
) -> None:
    targets = [str(t) for t in (target_columns or []) if str(t).strip()]
    if len(targets) > 1 and not spec.capabilities.supports_multitask:
        raise ValueError(
            f"Model '{spec.name}' does not support multitask targets: {targets}. "
            "Select a single target or use a multitask-capable model."
        )
    required = set(spec.capabilities.input_requires)
    provided = {str(v).strip() for v in feature_inputs if str(v).strip()}
    missing = sorted(required - provided)
    if missing:
        provided_str = ", ".join(sorted(provided)) if provided else "none"
        raise ValueError(
            f"Model '{spec.name}' requires inputs {missing} but featureset provides {provided_str}. "
            "Update the featurizer/featureset or choose a compatible model."
        )
