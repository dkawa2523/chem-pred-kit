# Task 127 (P2): Uncertainty & Ensembling

## 目的
予測の信頼度推定（uncertainty）と、比較実験のためのアンサンブル手法を追加する。

## In scope
- deep ensemble（同configでseed違いを複数run→平均/分散）
- MC dropout（可能なモデルのみ）
- 予測CSVに `y_std` 等を追加（artifact契約を拡張する場合は慎重に）

## Acceptance Criteria
- [ ] leaderboardに uncertainty 指標を追加できる
