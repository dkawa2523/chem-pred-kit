from __future__ import annotations

from src.tasks import resolve_target_columns, resolve_task


def test_resolve_target_columns_priority() -> None:
    cfg = {
        "task": {"target_columns": ["a"], "target_col": "b"},
        "data": {"target_col": "c"},
    }
    assert resolve_target_columns(cfg) == ["a"]


def test_resolve_target_columns_from_registry() -> None:
    cfg = {
        "target_registry": [
            {"name": "foo", "aliases": ["FOO"], "units": "K"},
        ],
        "task": {"target_names": ["foo"]},
    }
    assert resolve_target_columns(cfg) == ["target.foo"]


def test_resolve_task_metrics_filtered() -> None:
    cfg = {"task": {"target_columns": ["y"], "metrics": ["mae"]}}
    task = resolve_task(cfg)
    metrics = task.metrics_fn([1.0, 2.0], [1.0, 3.0])
    assert set(metrics.keys()) == {"mae"}


def test_resolve_task_loss_precedence() -> None:
    cfg = {"task": {"target_columns": ["y"], "loss": "mse"}, "train": {"loss": "huber"}}
    task = resolve_task(cfg)
    assert task.loss_name == "huber"
