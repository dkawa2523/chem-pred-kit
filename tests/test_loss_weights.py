from __future__ import annotations

import pytest

from src.common.losses import resolve_loss_weights


def test_resolve_loss_weights_list() -> None:
    cfg = {"loss": {"weights": [1.0, 2.5]}}
    assert resolve_loss_weights(cfg, ["a", "b"]) == [1.0, 2.5]


def test_resolve_loss_weights_dict_with_prefix() -> None:
    cfg = {"loss": {"weights": {"a": 0.5, "target.b": 2.0}}}
    assert resolve_loss_weights(cfg, ["a", "b"]) == [0.5, 2.0]


def test_resolve_loss_weights_missing_defaults() -> None:
    cfg = {"loss": {"weights": {"a": 0.5}}}
    assert resolve_loss_weights(cfg, ["a", "b"]) == [0.5, 1.0]


def test_resolve_loss_weights_length_mismatch() -> None:
    cfg = {"loss": {"weights": [1.0]}}
    with pytest.raises(ValueError):
        resolve_loss_weights(cfg, ["a", "b"])
