from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.utils import get_logger
from src.data.adapters.registry import create_dataset_adapter
from src.utils.artifacts import load_meta, validate_dataset_artifacts


def _write_raw_csv(path: Path) -> None:
    rows = [
        "CAS,MolecularFormula,Tc [K],Pc [Pa],Tb [K]",
        "A,CH4,190,4599000,111",
        "B,C2H6,305,4880000,184",
        "C,C3H8,370,4248000,231",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_lj_adapter_builds_dataset_artifacts(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    _write_raw_csv(raw_csv)

    out_dir = tmp_path / "processed"
    out_csv = out_dir / "dataset_with_lj.csv"
    out_indices_dir = out_dir / "indices"

    cfg = {
        "process": {"name": "build_dataset"},
        "target_registry": [
            {"name": "lj_epsilon_over_k_K", "aliases": ["lj_epsilon_over_k_K"], "units": "K"},
            {"name": "lj_sigma_A", "aliases": ["lj_sigma_A"], "units": "A"},
        ],
        "dataset": {
            "name": "lj_smoke",
            "units": {"lj_epsilon_over_k_K": "K", "lj_sigma_A": "A"},
        },
        "paths": {
            "raw_csv": str(raw_csv),
            "sdf_dir": str(tmp_path / "missing_sdf"),
            "out_csv": str(out_csv),
            "out_indices_dir": str(out_indices_dir),
        },
        "columns": {
            "cas": "CAS",
            "formula": "MolecularFormula",
            "tc": "Tc [K]",
            "pc": "Pc [Pa]",
            "tb": "Tb [K]",
        },
        "seed": 123,
        "lj": {
            "epsilon_method": "bird_critical",
            "sigma_method": "bird_critical",
            "epsilon_col": "lj_epsilon_over_k_K",
            "sigma_col": "lj_sigma_A",
            "valid_range": {"epsilon_over_k_K": [1.0, 5000.0], "sigma_A": [1.0, 20.0]},
        },
        "filter_invalid_lj": True,
        "selectors": [],
        "split": {"method": "random", "seed": 123, "fractions": [0.6, 0.2, 0.2]},
    }

    logger = get_logger("test_dataset_adapter_contract")
    adapter = create_dataset_adapter("lj_smoke")
    artifacts = adapter.build_processed(cfg, logger)

    assert artifacts.dataset_csv.exists()
    assert artifacts.indices_dir.exists()
    assert artifacts.dataset_index.exists()
    assert artifacts.meta_path.exists()

    df = pd.read_csv(artifacts.dataset_csv)
    assert "target.lj_epsilon_over_k_K" in df.columns
    assert "target.lj_sigma_A" in df.columns

    validate_dataset_artifacts(artifacts.meta_path.parent)
    meta = load_meta(artifacts.meta_path.parent)
    assert meta["dataset_hash"] == artifacts.dataset_hash
    assert meta["target_names"] == ["lj_epsilon_over_k_K", "lj_sigma_A"]
    assert meta["units"]["lj_epsilon_over_k_K"] == "K"
    assert meta["units"]["lj_sigma_A"] == "A"
