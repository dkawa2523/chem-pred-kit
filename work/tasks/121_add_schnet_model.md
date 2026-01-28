# Task 121 (P0): SchNet Integration

## 目的
3D距離ベースモデル SchNet を導入し、QM9/MD17で学習できるようにする。

## 実装方針
- PyG built-in SchNet が利用できるならそれを優先
- 依存が厳しい場合は optional dependency として torchmd-net も候補に残す

## In scope
- model=schnet の追加（registry + config）
- required: pos, radius graph, distance features
- multitask head は Task 130 に従う（まずは単一ターゲットでOK）
- smoke test（QM9 smallでtrain→eval）

## Acceptance Criteria
- [x] `model=schnet` で QM9 gap が学習できる
- [x] artifact契約が保たれる
