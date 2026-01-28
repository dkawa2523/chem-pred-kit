from __future__ import annotations

from pathlib import Path

import numpy as np

from src.targets.transform import TargetTransform, load_target_transforms, save_target_transforms


def test_target_transform_roundtrip() -> None:
    y = np.array([1.0, 2.0, 3.5, 5.0])
    transform = TargetTransform.from_config({"steps": ["log1p", "zscore"]})
    transform.fit(y)
    y_t = transform.transform(y)
    y_inv = transform.inverse_transform(y_t)
    assert np.allclose(y, y_inv, atol=1e-6)


def test_target_transform_clip() -> None:
    y = np.array([-1.0, 0.5, 2.0, 10.0])
    transform = TargetTransform.from_config({"steps": [{"name": "clip", "min": 0.0, "max": 2.0}]})
    transform.fit(y)
    y_t = transform.transform(y)
    assert np.all(y_t >= 0.0)
    assert np.all(y_t <= 2.0)


def test_target_transform_save_load(tmp_path: Path) -> None:
    y = np.array([0.5, 1.5, 2.5])
    transform = TargetTransform.from_config({"steps": ["log1p", "zscore"]})
    transform.fit(y)
    path = tmp_path / "target_transform.json"
    save_target_transforms(path, {"foo": transform})
    loaded = load_target_transforms(path)
    assert "foo" in loaded
    roundtrip = loaded["foo"].inverse_transform(loaded["foo"].transform(y))
    assert np.allclose(y, roundtrip, atol=1e-6)
