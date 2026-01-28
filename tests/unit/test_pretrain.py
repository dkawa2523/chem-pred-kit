import pytest


torch = pytest.importorskip("torch")

from torch import nn

from src.common.pretrain import apply_pretrain_freeze, build_pretrain_metadata, load_pretrained_weights


class DummyModel(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.head = nn.Linear(8, out_dim)

    def forward(self, x):
        return self.head(self.encoder(x))


def test_load_pretrained_weights_skips_mismatched_head(tmp_path):
    torch.manual_seed(0)
    pretrain = DummyModel(out_dim=1)
    nn.init.constant_(pretrain.encoder.weight, 0.5)
    nn.init.constant_(pretrain.encoder.bias, 0.1)
    nn.init.constant_(pretrain.head.weight, 1.5)
    nn.init.constant_(pretrain.head.bias, -0.2)

    ckpt_path = tmp_path / "model.ckpt"
    torch.save(pretrain.state_dict(), ckpt_path)

    downstream = DummyModel(out_dim=2)
    meta = build_pretrain_metadata({"ckpt_path": str(ckpt_path), "mode": "finetune"})
    report = load_pretrained_weights(downstream, meta, pretrain_cfg={"strict": False})

    assert torch.allclose(downstream.encoder.weight, pretrain.encoder.weight)
    assert torch.allclose(downstream.encoder.bias, pretrain.encoder.bias)
    assert any("head" in key for key in report.skipped_keys)


def test_apply_pretrain_freeze_linear_probe():
    model = DummyModel(out_dim=1)
    report = apply_pretrain_freeze(model, {"mode": "linear_probe"})

    assert report.trainable_params > 0
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_apply_pretrain_freeze_partial_freeze_patterns():
    model = DummyModel(out_dim=1)
    report = apply_pretrain_freeze(model, {"mode": "partial_freeze", "freeze_patterns": ["encoder"]})

    assert report.frozen_params > 0
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
