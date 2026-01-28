# Task 123 (P1): PaiNN Integration（E(3) equivariant）

## 目的
等変性モデル PaiNN を導入し、3D表現力を上げる。

## 方針
- torchmd-net 等の実装を利用する可能性が高い
- optional dependency とし、base環境は壊さない

## In scope
- model=painn の追加（registry + config）
- required: pos + neighbor graph
- smoke（QM9 small / MD17 energy）

## Acceptance Criteria
- [x] 依存が無い場合は丁寧なエラー
- [x] 依存がある場合は train→eval が通る
