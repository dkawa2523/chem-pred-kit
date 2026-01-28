import numpy as np
import pytest

from src.gnn.featurizer_graph import GraphFeaturizerConfig, featurize_mol_to_pyg


def test_angle_features_basic() -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCC")
    assert mol is not None

    pos = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    cfg = GraphFeaturizerConfig(
        node_features=["atomic_num"],
        edge_features=[],
        edge_mode="bond",
        radius_cutoff=2.0,
        max_neighbors=8,
        angle_features=["angle"],
        use_3d_pos=True,
    )
    data = featurize_mol_to_pyg(mol, y=None, cfg=cfg, pos=pos)

    assert hasattr(data, "angle_index")
    assert hasattr(data, "angles")
    assert data.angle_index.shape[0] == 3
    assert data.angles.shape[1] == 1

    angles = data.angles.detach().cpu().numpy().reshape(-1)
    assert angles.size == 2
    assert np.allclose(angles, np.pi / 2.0, atol=1e-5)
