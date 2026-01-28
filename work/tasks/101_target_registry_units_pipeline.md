# Task 101 (P0): Target Registry + Units/Transform Pipeline

## 目的
ターゲット名・単位・変換（log/standardize）を **設定とartifact**で一貫管理し、
データセットごとの差を吸収できるようにする。

## In scope
- `target registry`（canonical target名、別名マッピング、units）
- `TargetTransform`（fit→save→load→apply）
  - 例：log1p、z-score、clipping
- dataset adapterからのターゲット取り込み時に registry を通して `target.<name>` を生成
- meta/metrics に `target_names` と `units` と `transform` を記録

## Out of scope
- マルチタスク loss weighting（Task 131）

## Acceptance Criteria
- [ ] datasetごとに列名が違っても canonical 名で学習できる
- [ ] train/predict で同じ transform が使われる（skewなし）
- [ ] transform state が artifact として保存される

## Verification
- `python scripts/train.py task=... dataset=...`（既存の実行形式に合わせる）
- `pytest tests/contract -q`
