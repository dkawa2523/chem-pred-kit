# Task 125 (P1): Graphormer（2D Transformer）

## 目的
2D Transformer 系モデルを追加し、FP/GCN以外の強いベースラインを用意する。

## 実装方針
- PyGにGraphormer相当があればそれを利用
- 無い場合、依存を増やさずに実装できる簡易GraphTransformerを先に導入し、GraphormerはP2へ回す

## In scope
- model=graphormer（or graph_transformer）の追加
- required: 2D graph + positional encoding（shortest pathなど）
- smoke（QM9 small / MoleculeNet）

## Acceptance Criteria
- [x] 2Dモデルとして比較実験が可能

## Done
- Graphormer/Graph Transformer の registry + config 追加
- shortest-path attention bias の 2D Transformer baseline を実装
- MoleculeNet での smoke + config loader テスト追加

## Verification
- `pytest tests/test_config_loader.py::test_load_graphormer_model_config`
- `pytest tests/smoke/test_moleculenet_graphormer_smoke.py::test_moleculenet_graphormer_smoke_artifacts`
