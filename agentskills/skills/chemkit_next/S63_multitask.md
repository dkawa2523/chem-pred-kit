# S63: Multitask Learning

## 目的
- 多ターゲット回帰/分類を同一基盤で扱えるようにする

## 必須設計
- y: (batch, T)
- mask_y: (batch, T)（欠損はloss/metricから除外）
- metric: タスク別 + 全体平均

## よくある落とし穴
- view(-1) で潰して shape mismatch
- タスク間スケール差で一部だけ学習が支配
