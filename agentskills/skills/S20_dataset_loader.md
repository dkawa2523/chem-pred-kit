# S20 Dataset Loader

## Purpose
CSV/SDF 入力を共通表現へ変換するローダを追加/修正し、目的変数の差し替えに耐える基盤を作る。

## Inputs
- docs/02_DATA_CONTRACTS.md
- work/tasks/040_add_new_property_task.md（または該当）

## Allowed Changes
- src/data/**
- configs/dataset/**
- tests/**

## Steps
1) 現状の CSV/SDF の読み込み箇所を特定
2) column mapping を config で指定できるようにする
3) 欠損/不一致の扱いをログ化
4) unit test（最小CSV）を追加
