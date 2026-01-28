from __future__ import annotations

import pytest

from src.featuresets.registry import create_featureset
from src.models import (
    feature_inputs_from_featureset,
    get_model_spec,
    resolve_model_family,
    validate_model_requirements,
)


def test_resolve_model_family() -> None:
    assert resolve_model_family({"model": {"name": "mpnn"}}) == "gnn"
    assert resolve_model_family({"model": {"name": "lightgbm"}}) == "fp"
    assert resolve_model_family({"model": {"name": "chemberta"}}) == "smiles"


def test_model_requires_edge_attr() -> None:
    cfg = {
        "model": {"name": "mpnn"},
        "featurizer": {"node_features": ["atomic_num"], "edge_features": []},
    }
    featureset = create_featureset(cfg)
    spec = get_model_spec("mpnn")
    feature_inputs = feature_inputs_from_featureset(featureset)
    with pytest.raises(ValueError, match="edge_attr"):
        validate_model_requirements(spec, ["target.lj_sigma"], feature_inputs)


def test_model_rejects_multitask_targets() -> None:
    spec = get_model_spec("lightgbm")
    with pytest.raises(ValueError, match="multitask"):
        validate_model_requirements(spec, ["target.a", "target.b"], set())


def test_smiles_model_requires_tokens() -> None:
    spec = get_model_spec("chemberta")
    assert "tokens" in spec.capabilities.input_requires


def test_model_requires_angles() -> None:
    cfg = {
        "model": {"name": "dimenetpp"},
        "featurizer": {"node_features": ["atomic_num"], "edge_features": [], "use_3d_pos": True},
    }
    featureset = create_featureset(cfg)
    spec = get_model_spec("dimenetpp")
    feature_inputs = feature_inputs_from_featureset(featureset)
    with pytest.raises(ValueError, match="angles"):
        validate_model_requirements(spec, ["target.lj_sigma"], feature_inputs)


def test_model_requires_pos() -> None:
    cfg = {
        "model": {"name": "egnn"},
        "featurizer": {"node_features": ["atomic_num"], "edge_features": [], "use_3d_pos": False},
    }
    featureset = create_featureset(cfg)
    spec = get_model_spec("egnn")
    feature_inputs = feature_inputs_from_featureset(featureset)
    with pytest.raises(ValueError, match="pos"):
        validate_model_requirements(spec, ["target.lj_sigma"], feature_inputs)
