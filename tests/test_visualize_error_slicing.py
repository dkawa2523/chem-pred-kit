from __future__ import annotations

import pandas as pd

from src.common.metrics import regression_metrics
from src.visualize.error_slicing import ErrorSlicingConfig, build_error_slicing_table
from src.visualize.utils import resolve_plot_targets


def test_build_error_slicing_table_basic() -> None:
    pred_df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.2, 1.9, 2.7, 5.1],
            "split": ["val", "val", "test", "test"],
        }
    )
    feat_df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "n_atoms": [5, 6, 12, 15],
            "n_heavy_atoms": [3, 4, 7, 8],
            "ring_count": [0, 1, 2, 1],
            "rotatable_bonds": [0, 2, 1, 3],
            "_elements_list": [["C", "H", "O"], ["C", "H", "O"], ["C", "H", "N"], ["C", "H", "N"]],
        }
    )
    merged = pred_df.merge(feat_df, on="sample_id", how="left")

    targets, _ = resolve_plot_targets(merged.columns, target_mode="primary", primary_target=None)
    slice_cfg = ErrorSlicingConfig(
        splits=["val", "test"],
        min_samples=2,
        target_mode="primary",
        element_slices=["O", "N", "Cl"],
        include_halogen=True,
        size_bins={"n_atoms": [0, 10, 20, 40], "n_heavy_atoms": [0, 5, 10, 20]},
        ring_bins=[0, 1, 2, 3],
        rot_bonds_bins=[0, 1, 2, 4],
    )

    table = build_error_slicing_table(merged, targets, regression_metrics, slice_cfg)

    assert not table.empty
    assert "slice_type" in table.columns
    assert (table["slice_key"] == "n_atoms").any()
    assert (table["slice_key"] == "element_O").any()
    assert set(table["split"].unique()).issubset({"val", "test"})
