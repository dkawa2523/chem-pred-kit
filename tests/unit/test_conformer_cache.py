from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.conformer import ConformerCache, ConformerConfig, ConformerService, mol_from_smiles


def test_conformer_cache_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    cfg = ConformerConfig(
        seed=11,
        forcefield="uff",
        max_iters=10,
        on_failure="retry",
        max_attempts=2,
        num_conformers=3,
        aggregation="mean",
    )
    cache_path = tmp_path / "conformer_cache.json"
    cache = ConformerCache(cache_path, config={"conformer": "test"})
    service = ConformerService(cfg, cache=cache)

    mol = mol_from_smiles("CCO")
    assert mol is not None
    pos, meta = service.get_pos(sample_id="sample-1", mol=mol, smiles="CCO", allow_generate=True)
    assert pos is not None
    assert pos.shape[0] == mol.GetNumAtoms()
    assert meta.get("method") is not None
    assert meta.get("aggregation") == "mean"
    assert meta.get("num_conformers") == 3
    assert meta.get("num_conformers_used", 1) >= 1

    cache.save()

    cache2 = ConformerCache(cache_path)
    service2 = ConformerService(cfg, cache=cache2)
    pos2, meta2 = service2.get_pos(sample_id="sample-1", mol=mol, smiles="CCO", allow_generate=False)
    assert pos2 is not None
    assert np.allclose(pos, pos2)
    assert meta2.get("status") in {"ok", "partial", "fallback_2d"}
    assert meta2.get("aggregation") == "mean"
    assert meta2.get("num_conformers") == 3
