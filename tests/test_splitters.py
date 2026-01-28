from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.splitters import (
    build_group_map,
    group_split,
    random_split,
    scaffold_split,
    time_split,
    validate_group_leakage,
    validate_scaffold_split,
    validate_split_indices,
    validate_time_split,
)


def test_random_split_reproducible() -> None:
    df = pd.DataFrame({"CAS": [f"X{i:02d}" for i in range(30)]})
    first = random_split(df, ratios=(0.7, 0.2, 0.1), seed=123)
    second = random_split(df, ratios=(0.7, 0.2, 0.1), seed=123)

    assert first == second
    validate_split_indices(first)


def test_group_split_keeps_groups() -> None:
    df = pd.DataFrame({"group": ["A", "A", "B", "B", "C", "C"]})
    indices = group_split(df, group_col="group", ratios=(0.34, 0.34, 0.32), seed=0)

    group_map = build_group_map(df, group_col="group")
    validate_group_leakage(indices, group_map, label="group")


def test_scaffold_split_no_leakage(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem

    def _write_sdf(path: Path, smiles: str) -> None:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()

    sdf_dir = tmp_path / "sdf"
    sdf_dir.mkdir()

    cas_values = ["A", "B", "C", "D", "E", "F"]
    smiles_list = [
        "c1ccccc1",
        "c1ccccc1O",
        "C1CCCCC1",
        "C1CCCCC1O",
        "n1ccccc1",
        "n1ccccc1O",
    ]
    for cas, smiles in zip(cas_values, smiles_list):
        _write_sdf(sdf_dir / f"{cas}.sdf", smiles)

    df = pd.DataFrame({"CAS": cas_values})
    indices = scaffold_split(df, sdf_dir=sdf_dir, cas_col="CAS", ratios=(0.34, 0.34, 0.32), seed=0)
    second = scaffold_split(df, sdf_dir=sdf_dir, cas_col="CAS", ratios=(0.34, 0.34, 0.32), seed=0)

    assert indices == second
    validate_split_indices(indices)
    validate_scaffold_split(df, sdf_dir=sdf_dir, cas_col="CAS", indices=indices)


def test_time_split_reproducible_and_ordered() -> None:
    df = pd.DataFrame({"time": [3, 1, 2, 5, 4]})
    first = time_split(df, time_col="time", ratios=(0.6, 0.2, 0.2), ascending=True)
    second = time_split(df, time_col="time", ratios=(0.6, 0.2, 0.2), ascending=True)

    assert first == second
    validate_time_split(df, time_col="time", indices=first, ascending=True)

    train_times = df.loc[first["train"], "time"]
    val_times = df.loc[first["val"], "time"]
    test_times = df.loc[first["test"], "time"]
    assert train_times.max() <= val_times.min()
    assert val_times.max() <= test_times.min()
