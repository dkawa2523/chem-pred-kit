from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from src.featuresets.base import (
    FeatureSetBase,
    GraphFeatureSet,
    SmilesFeatureSet,
    TabularFeatureSet,
    compute_featureset_hash,
    load_featureset_hash,
    load_featureset_manifest,
)
from src.utils.artifacts import resolve_featureset_name

FeatureSetClass = Type[FeatureSetBase]

_FEATURESET_CLASSES: Dict[str, FeatureSetClass] = {}


def register_featureset(name: str, cls: FeatureSetClass) -> None:
    _FEATURESET_CLASSES[str(name)] = cls


def _infer_featureset_class(cfg: Dict[str, Any]) -> FeatureSetClass:
    feat_cfg = cfg.get("featurizer", {}) if isinstance(cfg, dict) else {}
    if isinstance(feat_cfg, dict):
        name = str(feat_cfg.get("name", "")).lower()
        if name in {"smiles", "smiles_tokenizer", "smiles-transformer", "smiles_transformer"}:
            return SmilesFeatureSet
    if isinstance(feat_cfg, dict) and (feat_cfg.get("node_features") or feat_cfg.get("edge_features")):
        return GraphFeatureSet
    return TabularFeatureSet


def _ensure_registry_loaded() -> None:
    if _FEATURESET_CLASSES:
        return
    import importlib

    importlib.import_module("src.featuresets.builtin")


def create_featureset(cfg: Dict[str, Any]) -> FeatureSetBase:
    _ensure_registry_loaded()
    name = resolve_featureset_name(cfg)
    if not name:
        raise ValueError("featureset name could not be resolved from config.")
    cls = _FEATURESET_CLASSES.get(name) or _infer_featureset_class(cfg)
    return cls.from_config(name=name, cfg=cfg)


def load_featureset(artifacts_dir: Path, cfg: Optional[Dict[str, Any]]) -> FeatureSetBase:
    _ensure_registry_loaded()
    manifest = load_featureset_manifest(artifacts_dir)
    name = resolve_featureset_name(cfg or {}) if cfg is not None else None
    cfg_for_load: Dict[str, Any] = cfg or {}
    expected_hash = load_featureset_hash(artifacts_dir)

    if manifest:
        if manifest.get("name"):
            name = manifest.get("name")
        manifest_cfg = manifest.get("config")
        if isinstance(manifest_cfg, dict):
            cfg_for_load = manifest_cfg
        if manifest.get("featureset_hash"):
            expected_hash = str(manifest.get("featureset_hash"))

    if not name:
        raise ValueError("featureset name could not be resolved for loading.")

    cls = _FEATURESET_CLASSES.get(name) or _infer_featureset_class(cfg_for_load)
    featureset = cls.load(name=name, artifacts_dir=Path(artifacts_dir), cfg=cfg_for_load)

    if manifest and manifest.get("feature_meta"):
        featureset.feature_meta = manifest.get("feature_meta")

    if expected_hash:
        actual_hash = compute_featureset_hash(featureset.manifest(), featureset.artifact_paths(Path(artifacts_dir)))
        if actual_hash != expected_hash:
            raise ValueError(
                f"featureset hash mismatch (expected {expected_hash}, got {actual_hash})."
            )

    return featureset
