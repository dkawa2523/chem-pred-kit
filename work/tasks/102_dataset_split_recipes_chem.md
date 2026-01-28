# Task 102 (P0): Chem Datasets Split Recipes（QM9/MD17/MoleculeNet）

## 目的
データセットごとの「リークしない split」を、configで再現可能にする。

## In scope
- split strategy の拡張/整理（random/scaffold/time/group）
- datasetごとの推奨split recipe を docs/specs に追記
- `split artifact` の保存と再利用（train/eval/predictが同じsplitを参照）
- MD17の time-based split を優先（時間相関によるリークを避ける）

## Acceptance Criteria
- [x] QM9のrandom splitが再現可能（seed固定）
- [x] MoleculeNetのscaffold splitが再現可能
- [x] MD17のtime splitが再現可能で、リーク検知が通る

## Verification
- `python scripts/build_dataset.py dataset=qm9 split=scaffold`（例）
- `python scripts/build_dataset.py dataset=md17 split=time`
