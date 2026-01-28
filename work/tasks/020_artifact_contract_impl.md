# Task 020 (P0): Artifact 契約をコードに実装

## 目的
- 「どのデータ/設定/コードで学習したモデルか」を必ず追跡できるようにする

## Plan
1) `src/utils/artifacts.py`（案）を作り、artifact 保存の共通関数を実装
2) 学習終了時に `config.yaml`, `meta.json`, `metrics.json`, `model.ckpt` を保存
3) 推論時に `predictions.csv` を保存し、meta に model_version 等を書き込む
4) contract test を追加（必須ファイル/キーの検査）

## Acceptance Criteria
- [x] docs/04 の必須成果物が生成される
- [x] contract test が通る

## Verification
- `pytest tests/contract/test_artifacts_contract.py`

## Notes
- Blocked note was stale; 010/015 are done so artifact paths and config composition are fixed.
