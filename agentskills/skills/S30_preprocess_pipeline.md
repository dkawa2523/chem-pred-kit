# S30 Preprocess Pipeline

## Purpose
Mol の前処理（sanitize、欠損処理、3D生成など）をパイプライン化し、学習/推論で共通化する。

## Inputs
- docs/00_INVARIANTS.md（skew禁止）
- work/tasks/030_featurepipeline_unify.md（または該当）

## Allowed Changes
- src/common/** or src/*/preprocess/**
- configs/preprocess/**
- tests/**

## Common Pitfalls
- 推論だけ別の前処理をしてしまう
- 3D生成の乱数・再現性を考慮しない
