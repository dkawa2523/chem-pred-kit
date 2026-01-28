from __future__ import annotations

from src.visualize.utils import build_metrics_tables, resolve_plot_targets, sanitize_filename


def test_resolve_plot_targets_primary_multitask() -> None:
    columns = ["y_true_alpha", "y_pred_alpha", "y_true_beta", "y_pred_beta"]
    targets, available = resolve_plot_targets(columns, target_mode="primary", primary_target="beta")

    assert available == ["alpha", "beta"]
    assert len(targets) == 1
    assert targets[0].name == "beta"
    assert targets[0].y_true_col == "y_true_beta"
    assert targets[0].y_pred_col == "y_pred_beta"
    assert targets[0].suffix == ""


def test_resolve_plot_targets_all_multitask() -> None:
    columns = ["y_true_alpha", "y_pred_alpha", "y_true_beta", "y_pred_beta"]
    targets, available = resolve_plot_targets(columns, target_mode="all", primary_target=None)

    assert available == ["alpha", "beta"]
    assert [t.name for t in targets] == ["alpha", "beta"]
    assert [t.suffix for t in targets] == ["_alpha", "_beta"]


def test_resolve_plot_targets_fallback() -> None:
    columns = ["y_true", "y_pred"]
    targets, available = resolve_plot_targets(columns, target_mode="all", primary_target=None)

    assert available == []
    assert len(targets) == 1
    assert targets[0].y_true_col == "y_true"
    assert targets[0].y_pred_col == "y_pred"


def test_sanitize_filename() -> None:
    assert sanitize_filename("alpha/beta") == "alpha_beta"
    assert sanitize_filename(" target ") == "target"


def test_build_metrics_tables() -> None:
    metrics = {
        "by_split": {
            "train": {
                "mae": 0.1,
                "rmse": 0.2,
                "by_target": {"alpha": {"mae": 0.11}, "beta": {"mae": 0.22}},
            },
            "val": {"mae": 0.3},
        }
    }

    split_table, target_table = build_metrics_tables(metrics)

    assert set(split_table["split"].tolist()) == {"train", "val"}
    assert "mae" in split_table.columns
    assert target_table is not None
    assert set(target_table["target"].tolist()) == {"alpha", "beta"}
