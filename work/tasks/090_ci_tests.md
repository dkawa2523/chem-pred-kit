# Task 090 (P2): CI/テスト整備

## 目的
- 何十回の改修でも壊れないように、自動テストとlintを入れる

## Plan
1) `pytest` の導入と最小テスト追加
2) smoke config（quick）で train/predict が通る integration test
3) GitHub Actions（任意）で test を回す

## Acceptance Criteria
- [x] `pytest -q` が通る
- [x] integration smoke が通る

## Notes
- GitHub Actions で pytest を実行する workflow を追加済み
