import pytest

from src.common.feature_pipeline import GraphFeaturePipeline
from src.gnn.featurizer_graph import GraphFeaturizerConfig, featurize_mol_to_pyg


def test_graph_config_adds_partial_charge_features() -> None:
    cfg = {
        "featurizer": {"node_features": ["atomic_num"], "edge_features": []},
        "features": {"add_partial_charge": True},
    }
    pipeline = GraphFeaturePipeline.from_config(cfg)
    node_features = pipeline.graph_cfg.node_features
    assert "partial_charge" in node_features
    assert "partial_charge_mask" in node_features


def test_partial_charge_node_feature_mask() -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    import torch
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    cfg = GraphFeaturizerConfig(
        node_features=["atomic_num", "partial_charge", "partial_charge_mask"],
        edge_features=[],
    )
    data = featurize_mol_to_pyg(mol, y=None, cfg=cfg)
    assert data.x.shape[1] == 3

    charges = data.x[:, 1]
    mask = data.x[:, 2]
    assert torch.all(torch.isfinite(charges))
    assert torch.all((mask == 0) | (mask == 1))
    assert torch.any(mask == 1)
