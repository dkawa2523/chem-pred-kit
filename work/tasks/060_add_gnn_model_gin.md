# Task 060 (P1): GNN に新モデル（例: GIN）を追加

## 目的
- モデル追加の型を確立し、今後のモデル拡張を高速化する

## Plan
1) `src/gnn/models.py`（実在するなら）に GIN を追加（なければ適切な場所へ）
2) `configs/model/gin.yaml` を追加し、`model.name: gin` で選べるようにする
3) 最小 smoke: 1 epoch で学習・推論が通る
4) 学習ログ/成果物が契約どおり出る

## Acceptance Criteria
- [x] `model.name=gin` で学習が通る
- [x] predict が通る
