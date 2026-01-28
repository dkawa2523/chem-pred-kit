from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.featuresets.base import load_featureset_hash
from src.featuresets.registry import create_featureset, load_featureset
from src.common.io import load_sdf_mol, sdf_path_from_cas


def _load_fixture_mol(cas: str):
    root = Path(__file__).resolve().parents[2]
    sdf_dir = root / "tests" / "fixtures" / "data" / "raw" / "sdf_files"
    return load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))


def _base_cfg() -> dict:
    return {
        "featurizer": {
            "fingerprint": "morgan",
            "morgan_radius": 2,
            "n_bits": 128,
            "use_counts": False,
        },
        "preprocess": {"impute_nan": "mean", "standardize": True},
    }


def test_featureset_roundtrip_hash(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")

    cfg = _base_cfg()
    featureset = create_featureset(cfg)

    mol_a = _load_fixture_mol("64-17-5")
    mol_b = _load_fixture_mol("67-64-1")
    assert mol_a is not None
    assert mol_b is not None

    x1, _ = featureset.pipeline.featurize_mol(mol_a)
    x2, _ = featureset.pipeline.featurize_mol(mol_b)
    X_train = np.vstack([x1, x2]).astype(float)
    X_train[0, 0] = np.nan

    featureset.fit(X_train)
    x1_t, meta1 = featureset.transform_mol(mol_a)

    featureset_hash = featureset.save(tmp_path)
    assert load_featureset_hash(tmp_path) == featureset_hash

    loaded = load_featureset(tmp_path, cfg)
    x1_t_loaded, meta2 = loaded.transform_mol(mol_a)
    assert np.allclose(x1_t, x1_t_loaded, equal_nan=True)
    assert meta1["fp_type"] == meta2["fp_type"]


def test_featureset_hash_mismatch_raises(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")

    cfg = _base_cfg()
    featureset = create_featureset(cfg)

    mol = _load_fixture_mol("64-17-5")
    assert mol is not None
    x1, _ = featureset.pipeline.featurize_mol(mol)
    featureset.fit(np.vstack([x1, x1]).astype(float))
    featureset.save(tmp_path)

    manifest_path = tmp_path / "featureset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["featureset_hash"] = "deadbeef"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="featureset hash mismatch"):
        load_featureset(tmp_path, cfg)
