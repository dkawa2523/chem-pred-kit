from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.common.losses import masked_regression_loss


def test_masked_loss_ignores_missing() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [3.0, 10.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    loss = masked_regression_loss(pred, target, mask=mask, loss_name="mse")
    assert loss.item() == pytest.approx(12.0)


def test_masked_loss_with_weights() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [3.0, 10.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    weights = [1.0, 2.0]

    loss = masked_regression_loss(pred, target, mask=mask, loss_name="mse", weights=weights)
    assert loss.item() == pytest.approx(24.0)
