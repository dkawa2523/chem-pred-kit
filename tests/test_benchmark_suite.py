from __future__ import annotations

import json
from pathlib import Path

from src.common.benchmark_suite import run


def test_benchmark_suite_dry_run_creates_plan(tmp_path: Path) -> None:
    cfg = {
        "process": {"name": "benchmark_suite"},
        "benchmark_suite": {
            "name": "unit_suite",
            "matrix": {
                "datasets": ["qm9"],
                "targets": ["gap"],
                "entries": [
                    {"name": "fp", "model": "fp_lightgbm", "features": "fp_morgan_desc"},
                ],
            },
            "run": {"dry_run": True, "evaluate": False, "visualize": False, "leaderboard": False},
        },
        "output": {"run_dir": str(tmp_path / "runs"), "exp_name": "unit_suite"},
    }

    run_dir = run(cfg)
    plan_path = run_dir / "benchmark_plan.json"
    report_path = run_dir / "benchmark_report.md"

    assert plan_path.exists()
    assert report_path.exists()

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert "cases" in plan
    assert len(plan["cases"]) == 1
    case = plan["cases"][0]
    assert case["dataset"] == "qm9"
    assert case["target"] == "gap"

    report = report_path.read_text(encoding="utf-8")
    assert "## Convergence" in report
    assert "|qm9_gap_fp|dry_run|" in report
