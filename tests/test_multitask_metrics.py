from __future__ import annotations

import numpy as np
import pytest

from src.tasks import resolve_task


def test_multitask_metrics_with_mask() -> None:
    cfg = {"task": {"target_columns": ["target.a", "target.b"], "metrics": "regression"}}
    task = resolve_task(cfg)

    y_true = np.array([[1.0, np.nan], [2.0, 4.0], [np.nan, 5.0]], dtype=float)
    y_pred = np.array([[1.1, 0.0], [1.9, 4.2], [0.0, 4.8]], dtype=float)
    mask = np.isfinite(y_true)

    metrics = task.metrics_fn(y_true, y_pred, mask)

    assert "by_target" in metrics
    assert set(metrics["by_target"].keys()) == {"a", "b"}
    assert metrics["by_target"]["a"]["mae"] == pytest.approx(0.1)
    assert metrics["by_target"]["b"]["mae"] == pytest.approx(0.2)
    assert metrics["rmse"] == pytest.approx(0.15)
