# Task 162 (P1): Optional Dependencies CI Strategy

## 目的
optional dependency を導入しても、base CI を壊さず、必要な範囲で検証できるようにする。

## In scope
- extras導入（3d/transformer/pretrain） or requirements分離
- base CI: FP/GCN + QM9 fixture まで
- optional CI: SchNet/DimeNet++/PaiNN/Transformer を夜間 or 手動で実行

## Acceptance Criteria
- [ ] base CI が軽く安定
- [ ] optional CI で追加モデルが検証できる
