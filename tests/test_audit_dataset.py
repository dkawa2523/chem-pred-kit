from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem

from src.data.audit import audit_dataset


def _write_sdf(path: Path, smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def test_audit_dataset_basic(tmp_path: Path) -> None:
    sdf_dir = tmp_path / "sdf"
    sdf_dir.mkdir()

    _write_sdf(sdf_dir / "A.sdf", "CCO")
    _write_sdf(sdf_dir / "B.sdf", "CCO")

    df = pd.DataFrame(
        {
            "CAS": ["A", "B", "C"],
            "MolecularFormula": ["C2H6O", "C2H6O", "CH4"],
            "lj_epsilon_over_k_K": [100.0, 120.0, np.nan],
        }
    )
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)

    indices_dir = tmp_path / "indices"
    indices_dir.mkdir()
    (indices_dir / "train.txt").write_text("0\n", encoding="utf-8")
    (indices_dir / "test.txt").write_text("1\n", encoding="utf-8")
    (indices_dir / "val.txt").write_text("2\n", encoding="utf-8")

    cfg = {
        "audit": {
            "input": {"source": "processed"},
            "duplicate": {"methods": ["canonical_smiles"], "max_groups": 10, "max_examples": 5},
        },
        "data": {
            "dataset_csv": str(csv_path),
            "indices_dir": str(indices_dir),
            "sdf_dir": str(sdf_dir),
        },
        "task": {"target_col": "lj_epsilon_over_k_K"},
        "columns": {"cas": "CAS", "formula": "MolecularFormula"},
    }

    report, report_md, plot_data = audit_dataset(cfg)

    assert report["invalid_mol_count"] == 1
    assert isinstance(report.get("dataset_hash"), str)
    assert len(report["dataset_hash"]) == 64
    assert "canonical_smiles" in report["duplicate_groups"]
    assert report["target_stats"]["available"] is True
    leakage = report["split_leakage"]["by_method"]["canonical_smiles"]["leakage_group_count"]
    assert leakage == 1
