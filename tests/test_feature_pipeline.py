from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.feature_pipeline import EmbeddingFeaturePipeline, FingerprintFeaturePipeline, load_feature_pipeline, save_feature_pipeline
from src.common.io import load_sdf_mol, sdf_path_from_cas


def _load_fixture_mol(cas: str):
    root = Path(__file__).resolve().parents[1]
    sdf_dir = root / "tests" / "fixtures" / "data" / "raw" / "sdf_files"
    return load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))


def test_fp_pipeline_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    cfg = {
        "featurizer": {
            "fingerprint": "morgan",
            "morgan_radius": 2,
            "n_bits": 256,
            "use_counts": False,
        },
        "preprocess": {"impute_nan": "mean", "standardize": True},
    }

    pipeline = FingerprintFeaturePipeline.from_config(cfg)
    mol_a = _load_fixture_mol("64-17-5")
    mol_b = _load_fixture_mol("67-64-1")
    assert mol_a is not None
    assert mol_b is not None

    x1, _ = pipeline.featurize_mol(mol_a)
    x2, _ = pipeline.featurize_mol(mol_b)
    X_train = np.vstack([x1, x2]).astype(float)
    X_train[0, 0] = np.nan

    pipeline.fit(X_train)
    x1_t, meta1 = pipeline.transform_mol(mol_a)

    save_feature_pipeline(pipeline, tmp_path)
    loaded = load_feature_pipeline(tmp_path)
    assert isinstance(loaded, FingerprintFeaturePipeline)

    x1_t_loaded, meta2 = loaded.transform_mol(mol_a)
    assert np.allclose(x1_t, x1_t_loaded, equal_nan=True)
    assert meta1["fp_type"] == meta2["fp_type"]


def test_embedding_pipeline_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    cfg = {
        "featurizer": {
            "name": "pretrained_embedding",
            "backend": "stub",
            "embedding_dim": 32,
            "seed": 7,
            "normalize": True,
        },
        "preprocess": {"impute_nan": "mean", "standardize": True},
    }

    pipeline = EmbeddingFeaturePipeline.from_config(cfg)
    mol_a = _load_fixture_mol("64-17-5")
    mol_b = _load_fixture_mol("67-64-1")
    assert mol_a is not None
    assert mol_b is not None

    x1, _ = pipeline.featurize_mol(mol_a)
    x2, _ = pipeline.featurize_mol(mol_b)
    X_train = np.vstack([x1, x2]).astype(float)
    X_train[0, 0] = np.nan

    pipeline.fit(X_train)
    x1_t, meta1 = pipeline.transform_mol(mol_a)

    save_feature_pipeline(pipeline, tmp_path)
    loaded = load_feature_pipeline(tmp_path)
    assert isinstance(loaded, EmbeddingFeaturePipeline)

    x1_t_loaded, meta2 = loaded.transform_mol(mol_a)
    assert np.allclose(x1_t, x1_t_loaded, equal_nan=True)
    assert meta1["embedding_dim"] == meta2["embedding_dim"]
