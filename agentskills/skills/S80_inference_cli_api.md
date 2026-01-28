# S80 Inference CLI/API

## Purpose
推論 CLI を安定化し、将来 API 化できるように I/F を整える。

## Allowed Changes
- scripts/predict.py
- src/infer/**（提案）
- configs/infer/**

## Pitfalls
- モデルロード毎回で遅い
- 学習時と別の前処理


## Process分離（推奨）
- predict は推論のみに集中し、可視化は visualize Process へ分離する（肥大化防止）
- ClearML化を見越し、predict の meta.json に upstream（model run）の参照を残す
