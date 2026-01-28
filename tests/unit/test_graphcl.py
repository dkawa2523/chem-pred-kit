from __future__ import annotations

import pytest


def test_graphcl_loss_runs() -> None:
    torch = pytest.importorskip("torch")
    from src.gnn.graphcl import graphcl_loss

    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)
    loss = graphcl_loss(z1, z2, temperature=0.2)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_graphcl_augmentation_does_not_mutate() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from src.gnn.graphcl import GraphCLAugmentConfig, apply_graphcl_augmentation

    x = torch.ones((3, 2))
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.ones((4, 1))
    data = Data(x=x.clone(), edge_index=edge_index.clone(), edge_attr=edge_attr.clone())
    orig_x = data.x.clone()

    gen = torch.Generator().manual_seed(0)
    aug = apply_graphcl_augmentation(
        data,
        GraphCLAugmentConfig(edge_drop=0.5, node_feature_mask=0.5),
        generator=gen,
    )

    assert torch.allclose(data.x, orig_x)
    assert aug.x.shape == data.x.shape
    assert aug.edge_index.shape[0] == 2
    assert aug.edge_index.shape[1] <= data.edge_index.shape[1]
    if aug.edge_attr is not None:
        assert aug.edge_attr.shape[0] == aug.edge_index.shape[1]
