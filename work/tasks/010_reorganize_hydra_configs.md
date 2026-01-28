# Task 010 (P0): Hydra config を group 化して再整理

## 目的
- 現在の configs を用途別に整理し、設定の衝突/混乱を防ぐ
- モデル/特徴量/タスク/学習を組み替え可能にする

## Plan
1) `configs/` を group（dataset/task/preprocess/features/model/train/eval/infer）へ分割
2) 入口 `configs/config.yaml` を作り defaults を定義
3) 既存の学習スクリプトが新 config を読めるように薄い互換レイヤを用意
4) 最小 smoke 用 `dataset=quick` 相当の設定を用意

## Acceptance Criteria
- [ ] 既存の主要学習が新 config で動く（少なくとも 1 パス）
- [ ] config の合成結果が `runs/` に保存される
