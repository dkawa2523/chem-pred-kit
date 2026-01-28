from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_fixture_dataset_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "tests" / "fixtures" / "data" / "raw" / "tc_pc_tb_fixture.csv"
    sdf_dir = root / "tests" / "fixtures" / "data" / "raw" / "sdf_files"

    assert csv_path.exists()
    assert sdf_dir.exists()

    df = pd.read_csv(csv_path)
    required_cols = {"CAS", "MolecularFormula", "Tc [K]", "Pc [Pa]", "Tb [K]"}
    assert required_cols.issubset(set(df.columns))

    missing = []
    for cas in df["CAS"].astype(str).tolist():
        if not (sdf_dir / f"{cas}.sdf").exists():
            missing.append(cas)
    assert not missing, f"Missing SDF files for CAS: {missing}"
