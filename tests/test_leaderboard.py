from __future__ import annotations

import csv
from pathlib import Path

from src.common.leaderboard import run
from src.common.utils import save_json


def _write_meta(run_dir: Path, **overrides) -> None:
    meta = {
        "run_id": overrides.get("run_id", "run"),
        "process_name": overrides.get("process_name", "train"),
        "created_at": overrides.get("created_at", "2025-01-01T00:00:00Z"),
        "git_sha": overrides.get("git_sha", "deadbeef"),
        "dataset_hash": overrides.get("dataset_hash", "dataset"),
        "config_hash": overrides.get("config_hash", "config"),
        "task_name": overrides.get("task_name", "lj_epsilon"),
        "model_name": overrides.get("model_name", "rf"),
        "featureset_name": overrides.get("featureset_name", "fp_morgan"),
        "upstream_artifacts": [],
        "tags": overrides.get("tags", []),
    }
    meta.update(overrides)
    save_json(run_dir / "meta.json", meta)


def _write_metrics(run_dir: Path, payload: dict) -> None:
    save_json(run_dir / "metrics.json", payload)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_leaderboard_collects_and_filters(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "train" / "run_a"
    run_a.mkdir(parents=True)
    _write_meta(run_a, run_id="run_a", process_name="train", model_name="rf")
    _write_metrics(
        run_a,
        {"val": {"r2": 0.7}, "test": {"r2": 0.6}, "n_train": 10, "n_val": 2, "n_test": 2, "seed": 0},
    )

    run_b = runs_root / "evaluate" / "run_b"
    run_b.mkdir(parents=True)
    _write_meta(run_b, run_id="run_b", process_name="evaluate", model_name="lgbm")
    _write_metrics(run_b, {"by_split": {"val": {"r2": 0.9}}, "n_train": 10, "n_val": 2, "n_test": 2})

    cfg_filtered = {
        "process": {"name": "leaderboard"},
        "leaderboard": {
            "root_dir": str(runs_root),
            "metric_key": "r2",
            "metric_split": "val",
            "sort_order": "desc",
            "top_n": 10,
            "filters": {"model_name": "rf"},
        },
        "output": {"run_dir": str(tmp_path / "out"), "exp_name": "filtered"},
    }
    out_dir = run(cfg_filtered)
    rows = _read_csv(out_dir / "leaderboard.csv")
    assert len(rows) == 1
    assert rows[0]["model_name"] == "rf"
    assert float(rows[0]["metric_value"]) == 0.7

    cfg_all = {
        "process": {"name": "leaderboard"},
        "leaderboard": {
            "root_dir": str(runs_root),
            "metric_key": "r2",
            "metric_split": "val",
            "sort_order": "desc",
            "top_n": 10,
            "filters": {},
        },
        "output": {"run_dir": str(tmp_path / "out"), "exp_name": "all"},
    }
    out_dir_all = run(cfg_all)
    rows_all = _read_csv(out_dir_all / "leaderboard.csv")
    assert len(rows_all) == 2
    assert float(rows_all[0]["metric_value"]) == 0.9
    assert rows_all[0]["metric_source"].startswith("by_split")


def test_leaderboard_by_target_tables(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "evaluate" / "run_a"
    run_a.mkdir(parents=True)
    _write_meta(run_a, run_id="run_a", process_name="evaluate", model_name="rf")
    _write_metrics(
        run_a,
        {
            "by_split": {
                "val": {
                    "mae": 0.12,
                    "by_target": {
                        "gap": {"mae": 0.12},
                        "homo": {"mae": 0.2},
                    },
                }
            }
        },
    )

    run_b = runs_root / "evaluate" / "run_b"
    run_b.mkdir(parents=True)
    _write_meta(run_b, run_id="run_b", process_name="evaluate", model_name="lgbm")
    _write_metrics(
        run_b,
        {
            "by_split": {
                "val": {
                    "mae": 0.08,
                    "by_target": {
                        "gap": {"mae": 0.05},
                        "homo": {"mae": 0.25},
                    },
                }
            }
        },
    )

    cfg = {
        "process": {"name": "leaderboard"},
        "leaderboard": {
            "root_dir": str(runs_root),
            "metric_key": "mae",
            "metric_split": "val",
            "sort_order": "asc",
            "top_n": 10,
            "target_mode": "all",
        },
        "output": {"run_dir": str(tmp_path / "out"), "exp_name": "targets"},
    }
    out_dir = run(cfg)
    target_csv = out_dir / "leaderboard_by_target.csv"
    assert target_csv.exists()

    rows = _read_csv(target_csv)
    gap_rows = [row for row in rows if row["target_name"] == "gap"]
    homo_rows = [row for row in rows if row["target_name"] == "homo"]

    assert len(gap_rows) == 2
    assert len(homo_rows) == 2
    assert float(gap_rows[0]["metric_value"]) == 0.05
    assert gap_rows[0]["run_id"] == "run_b"
    assert float(homo_rows[0]["metric_value"]) == 0.2
    assert homo_rows[0]["run_id"] == "run_a"
