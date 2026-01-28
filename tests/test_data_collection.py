from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_collection.runner import run


def test_collect_data_dummy(tmp_path: Path) -> None:
    cfg = {
        "process": {"name": "collect_data"},
        "output": {"run_dir": str(tmp_path / "runs"), "exp_name": "dummy_run"},
        "data_collection": {
            "query": {"identifiers": ["64-17-5"], "limit": 1},
            "required_columns": ["CAS", "MolecularFormula", "Tc [K]"],
            "sample_id_column": "sample_id",
            "cache": {"enabled": False},
            "output": {"raw_csv_name": "raw.csv", "sdf_dir_name": "sdf_files", "overwrite": True},
            "export": {"enabled": False},
        },
        "data_source": {
            "name": "dummy",
            "client": {
                "records": [
                    {
                        "cas": "64-17-5",
                        "formula": "C2H6O",
                        "tc_k": 514.0,
                    }
                ]
            },
            "formatter": {
                "column_map": {
                    "CAS": "cas",
                    "MolecularFormula": "formula",
                    "Tc [K]": "tc_k",
                }
            },
        },
    }

    run_dir = run(cfg)
    raw_csv = run_dir / "raw" / "raw.csv"

    assert raw_csv.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "meta.json").exists()

    df = pd.read_csv(raw_csv)
    assert df.loc[0, "CAS"] == "64-17-5"
    assert "sample_id" in df.columns
