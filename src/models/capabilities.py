from __future__ import annotations

from typing import Any

from src.gnn.featurizer_graph import requires_3d_pos
from src.models.registry import ModelSpec


def _normalize_mode(value: Any) -> str:
    if value is True:
        return "on"
    if value is False or value is None:
        return "off"
    key = str(value).strip().lower()
    if key in {"on", "true", "enable", "enabled"}:
        return "on"
    if key in {"warn", "warning"}:
        return "warn"
    return "off"


def apply_model_capabilities(cfg, model_spec: ModelSpec, gcfg, conformer_cfg, logger) -> None:
    preprocess_cfg = cfg.get("preprocess", {}) if isinstance(cfg, dict) else {}
    mode = _normalize_mode(preprocess_cfg.get("capability_auto", "warn"))
    if mode == "off":
        return
    auto_on = mode == "on"

    inputs = set(model_spec.capabilities.input_requires)
    model_requires_pos = "pos" in inputs
    features_need_pos = requires_3d_pos(gcfg)

    if model_requires_pos and not getattr(gcfg, "use_3d_pos", False):
        msg = "Model requires 3D positions but featurizer.use_3d_pos is False."
        if auto_on:
            gcfg.use_3d_pos = True
            conformer_cfg.enabled = True
            logger.warning("%s Auto-enabled use_3d_pos/conformer.", msg)
        else:
            logger.warning("%s (capability_auto=warn)", msg)

    if not model_requires_pos and getattr(gcfg, "use_3d_pos", False) and not features_need_pos:
        msg = "Featurizer generates 3D positions but model does not require them."
        if auto_on:
            gcfg.use_3d_pos = False
            conformer_cfg.enabled = False
            logger.warning("%s Auto-disabled use_3d_pos/conformer.", msg)
        else:
            logger.warning("%s (capability_auto=warn)", msg)

    accepts_globals = getattr(model_spec.capabilities, "accepts_global_descriptors", True)
    if not accepts_globals and getattr(gcfg, "add_global_descriptors", None):
        msg = "Model does not accept global descriptors but featurizer adds them."
        if auto_on:
            gcfg.add_global_descriptors = None
            logger.warning("%s Auto-disabled add_global_descriptors.", msg)
        else:
            logger.warning("%s (capability_auto=warn)", msg)
