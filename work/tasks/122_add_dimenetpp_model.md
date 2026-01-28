# Task 122 (P1): DimeNet++ Integration（角度特徴）

## 目的
角度情報を使う DimeNet++ を導入し、QM9で高精度を狙う。

## 依存/注意
- PyG の DimeNet/DimeNet++ 実装の有無・API差がある
- 角度/三体相互作用の入力が必要（feature pipeline 側の整備も）

## In scope
- 利用可能な実装の調査（PyG / DIG / 外部）→ 採用方針をADRに残す
- model=dimenetpp の追加（optional dependency でも可）
- 必要入力（pos, edge_index, angle features）を capability に宣言
- smoke（QM9 small）

## Acceptance Criteria
- [x] optional dependency が無い環境では明確なエラー
- [x] ある環境では train→eval が通る

## Done
- DimeNet++ を PyG backend で追加し、角度特徴と optional dependency エラーを実装
- QM9 向けの featureset/config と smoke を追加

## Verification
- `python -m pytest tests/test_model_registry.py`
- `python -m pytest tests/test_graph_angle_features.py`
