# Task 130 (P0): Multitask Core（y多次元 + mask_y + metrics）

## 目的
QM9の複数物性などを同時に学習できるようにする。

## In scope
- dataset adapter が複数 target を出せる（`y: (N,T)`）
- 欠損を `mask_y` で表現し、loss/metrics から除外
- model head を out_dim=T に対応
- metrics をタスク別に出力し、artifactに保存
- configで `task.targets=[gap,homo,lumo,...]` を指定できる

## Acceptance Criteria
- [x] QM9で `targets=[gap,homo,lumo]` を同時学習できる
- [x] 欠損混在でも学習が落ちない（テスト追加）
