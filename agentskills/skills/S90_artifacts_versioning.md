# S90 Artifacts & Versioning

## Purpose
成果物とメタ情報の保存・ロードを統一し、再現性を保証する。

## Inputs
- docs/04_ARTIFACTS_AND_VERSIONING.md
- work/tasks/020_artifact_contract_impl.md

## Allowed Changes
- src/utils/artifacts.py（追加）
- scripts/train.py / predict.py（保存呼び出し）
- tests/contract/**

## Pitfalls
- 必須キーの欠落
- dataset_hash の未更新
