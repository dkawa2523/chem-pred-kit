from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_qm9_smoke_pipeline(qm9_smoke_artifacts) -> None:
    dataset_csv = Path(qm9_smoke_artifacts["dataset_csv"])
    df = pd.read_csv(dataset_csv)

    assert df.shape[0] == 256
    assert "target.gap" in df.columns

    features_dir = Path(qm9_smoke_artifacts["features_dir"])
    cache_key = qm9_smoke_artifacts["cache_key"]
    assert (features_dir / f"fp_features_{cache_key}.npz").exists()
    assert (features_dir / "features_manifest.json").exists()

    assert Path(qm9_smoke_artifacts["train_run_dir"]).exists()
    assert Path(qm9_smoke_artifacts["eval_run_dir"]).exists()
