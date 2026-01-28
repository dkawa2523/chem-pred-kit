from __future__ import annotations

import numpy as np
import pytest

from src.common.metrics import merge_metrics, uncertainty_metrics


def test_uncertainty_metrics_single_target() -> None:
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])
    y_std = np.array([1.0, 2.0])

    metrics = uncertainty_metrics(y_true, y_pred, y_std, target_names=["t"])

    expected_nll = float(np.mean(0.5 * np.log(2.0 * np.pi * (y_std**2))))
    assert metrics["nll"] == pytest.approx(expected_nll)
    assert metrics["y_std_mean"] == pytest.approx(1.5)
    assert metrics["by_target"]["t"]["nll"] == pytest.approx(expected_nll)


def test_uncertainty_metrics_multitask_masked() -> None:
    y_true = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
    y_pred = np.array([[1.1, 2.1], [0.0, 2.9], [4.2, 0.0]])
    y_std = np.array([[0.5, 1.0], [1.0, 2.0], [0.5, 1.0]])
    mask = np.isfinite(y_true)

    metrics = uncertainty_metrics(y_true, y_pred, y_std, mask=mask, target_names=["a", "b"])

    assert metrics["by_target"]["a"]["y_std_mean"] == pytest.approx(0.5)
    assert metrics["by_target"]["b"]["y_std_mean"] == pytest.approx(1.5)


def test_merge_metrics_merges_by_target() -> None:
    base = {"rmse": 1.0, "by_target": {"a": {"rmse": 1.0}}}
    extra = {"nll": 2.0, "by_target": {"a": {"nll": 2.0}, "b": {"nll": 3.0}}}

    merged = merge_metrics(base, extra)

    assert merged["rmse"] == 1.0
    assert merged["nll"] == 2.0
    assert merged["by_target"]["a"]["rmse"] == 1.0
    assert merged["by_target"]["a"]["nll"] == 2.0
    assert merged["by_target"]["b"]["nll"] == 3.0
