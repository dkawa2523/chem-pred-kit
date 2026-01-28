# Task 110 (P0): FeatureSet Registry（train/infer skew排除）

## 目的
特徴量生成を FeatureSet として統一し、train/eval/predictで **同一実装/同一状態**を使う。

## In scope
- FeatureSet interface（fit/transform/save/load）
- feature registry（features.name -> builder）
- 既存FP/GNN特徴量を FeatureSet にまとめる（必要なら薄いラッパで）
- featureset_hash を artifact に残す

## Acceptance Criteria
- [x] train と predict で同じ FeatureSet をロードして使える
- [x] skew が出ない（contract test追加）
