# S10 Hydra Config

## Purpose
Hydra config を group 化し、設定の再利用と組み合わせを容易にする。

## When to Use
- config が増えて混乱してきた
- モデル/特徴量/タスクを切り替えたい

## Inputs
- docs/03_CONFIG_CONVENTIONS.md
- work/tasks/010_reorganize_hydra_configs.md

## Allowed Changes
- configs/**
- scripts/*（config読込部分の更新）
- src/utils/validate_config.py（追加）

## Common Pitfalls
- defaults の上書き順序ミス
- rename で互換性を破る（禁止）


## Process対応（必須）
- configs/process/ を導入し、各処理（train/evaluate/predict/visualize等）をHydraで独立設定できる形にする
- hydra.run.dir に process 名を含め、比較評価のrunが混ざらないようにする
