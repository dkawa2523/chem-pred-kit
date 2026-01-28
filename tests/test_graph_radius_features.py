import numpy as np
import pytest

from src.gnn.featurizer_graph import (
    GaussianExpansionConfig,
    GraphFeaturizerConfig,
    build_radius_graph,
    featurize_mol_to_pyg,
    gaussian_expansion,
)


def test_gaussian_expansion_centers() -> None:
    cfg = GaussianExpansionConfig(enabled=True, num_centers=4, min=0.0, max=3.0, sigma=1.0)
    distances = np.array([1.0], dtype=np.float32)
    out = gaussian_expansion(distances, cfg)
    assert out.shape == (1, 4)
    assert np.isclose(out[0, 1], 1.0, atol=1e-6)


def test_build_radius_graph_cutoff() -> None:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    edge_index, distances = build_radius_graph(pos, cutoff=1.5, max_neighbors=32)
    edges = {tuple(edge_index[:, i]) for i in range(edge_index.shape[1])}
    assert edges == {(0, 1), (1, 0)}
    assert np.allclose(sorted(distances.tolist()), [1.0, 1.0])


def test_radius_graph_distance_gaussian_edge_attr() -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    import torch
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CC")
    assert mol is not None

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    cfg = GraphFeaturizerConfig(
        node_features=["atomic_num"],
        edge_features=["distance", "gaussian"],
        edge_mode="radius",
        radius_cutoff=1.5,
        max_neighbors=4,
        gaussian_expansion=GaussianExpansionConfig(
            enabled=True,
            num_centers=4,
            min=0.0,
            max=2.0,
            sigma=0.5,
        ),
    )
    data = featurize_mol_to_pyg(mol, y=None, cfg=cfg, pos=pos)
    assert data.edge_index.shape == (2, 2)
    assert data.edge_attr.shape == (2, 5)
    assert data.pos.shape == (2, 3)
    assert torch.allclose(data.edge_attr[:, 0], torch.tensor([1.0, 1.0]), atol=1e-6)
