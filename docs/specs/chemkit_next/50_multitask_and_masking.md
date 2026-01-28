# Multitask & Masking（ChemKit Next）

## 目的
- QM9のような複数物性を「同時学習」できるようにする
- 欠損ターゲットが混在しても学習が壊れないようにする

## 設計
- encoder（共通） + head（taskごと or shared linear）
- loss は taskごとに計算し、mask_yで無効化できる

## loss weighting（段階導入）
- P0: uniform weight（単純平均）
- P1: 設定で重みを与えられる
- P2: 不確実性重み付け、GradNorm 等

## metrics
- 全体平均 + タスク別メトリクス（R2/MAE/RMSE）
- レポート生成でタスク別の表を出す
