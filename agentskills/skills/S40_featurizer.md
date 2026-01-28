# S40 Featurizer

## Purpose
特徴量（FP/記述子/3D/埋め込み）を追加し、設定で切替できるようにする。

## Inputs
- docs/02_DATA_CONTRACTS.md
- docs/03_CONFIG_CONVENTIONS.md
- work/tasks/030_featurepipeline_unify.md または NEW_FEATURIZER タスク

## Allowed Changes
- src/common/**（FeaturePipeline）
- src/fp/**, src/gnn/**（必要なら）
- configs/features/**
- tests/**

## Steps
1) featurizer I/F を確認（fit/transform or transformのみ）
2) 学習で fit した状態を artifact 保存
3) 推論で load して transform
4) 最小テスト追加

## Common Pitfalls
- “学習時に使ったスケーラー” を保存しない
- 推論で別実装を使ってしまう（skew）
