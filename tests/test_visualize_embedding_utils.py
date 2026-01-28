from __future__ import annotations

import numpy as np
import pandas as pd

from src.visualize.embedding import project_embeddings, resolve_color_values


def test_project_embeddings_pca_shape() -> None:
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, 5))
    coords = project_embeddings(embeddings, method="pca", random_state=42, params={})
    assert coords.shape == (12, 2)


def test_resolve_color_values_target_prefers_named() -> None:
    df = pd.DataFrame({"y_true_gap": [1.0, 2.0], "y_pred_gap": [1.1, 2.1]})
    values, label, col = resolve_color_values(
        df,
        color_by="target",
        color_target="gap",
        primary_target="gap",
    )
    assert label == "gap"
    assert col == "y_true_gap"
    assert values.tolist() == [1.0, 2.0]


def test_resolve_color_values_split() -> None:
    df = pd.DataFrame({"split": ["train", "val"]})
    values, label, col = resolve_color_values(
        df,
        color_by="split",
        color_target=None,
        primary_target=None,
    )
    assert label == "split"
    assert col == "split"
    assert values.tolist() == ["train", "val"]
