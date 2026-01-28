# Task 124 (P2): EGNN Integration（軽量equivariant）

## 目的
軽量な等変性モデル EGNN を導入し、3Dモデルの選択肢を増やす。

## In scope
- model=egnn の追加（optional dependencyの可能性）
- required: pos, edge_index（場合によって edge_attr）
- QM9 small で smoke

## Acceptance Criteria
- [x] SchNetと同条件で比較できる（configで切替）
