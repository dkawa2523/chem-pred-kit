# Task 151 (P1): Embedding Visualization（UMAP/t-SNE）

## 目的
モデル内部表現の可視化で、データセット差やエラー要因を分析できるようにする。

## In scope
- encoder embedding を抽出して2次元に射影（UMAP/t-SNE）
- ラベル（target, element count 等）で色分け（設定化）
- artifactに embeddings.csv と plot を保存

## Acceptance Criteria
- [x] QM9 で embedding plot が出る

## Verification
```sh
pytest -q tests/test_visualize_embedding_utils.py
```
