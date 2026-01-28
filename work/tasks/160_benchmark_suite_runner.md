# Task 160 (P0): Benchmark Suite Runner（組合せ実行）

## 目的
データセット×モデル×特徴量 の組合せを定義し、ベースライン比較を自動で回す。

## In scope
- benchmark matrix をYAMLで定義（dataset/model/features/targets）
- 逐次実行（autopilotと併用可）
- 結果を runs から集計し、leaderboardへ反映

## Acceptance Criteria
- [x] QM9 で baseline matrix（例：FP/GCN/SchNet）を回せる
- [x] 生成物が比較レポートになる
