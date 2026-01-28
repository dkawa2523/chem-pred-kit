# S70 Evaluator & Metrics

## Purpose
評価指標・可視化を追加し、比較可能性を高める。

## Allowed Changes
- src/common/metrics.py（存在するなら）または src/eval/**
- configs/eval/**
- tests/**

## Pitfalls
- metric 名の変更で互換を壊す
- CSV 出力の列を増やす時に契約更新を忘れる


## 比較評価（推奨）
- evaluate は `metrics.json` と `predictions.csv` を必ず出し、後から集計できる形にする
- 将来：複数runの集計（leaderboard）を別Processとして追加する
