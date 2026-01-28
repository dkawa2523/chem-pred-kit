# Task 141 (P1): Supervised Pretrain Recipe（PCQM4Mv2/QM9）

## 目的
教師あり pretrain の “最低限動くレシピ” を用意する。

## 方針
- 最初は QM9 の multitask pretrain（データ量が扱いやすい）
- 次に PCQM4Mv2（大規模）を optional で

## In scope
- pretrain config（epochs, batch, max_batches_per_epoch）
- pretrain→finetune のサンプル実験（1ターゲットでOK）
- 結果比較（収束速度/metric）をレポート化

## Acceptance Criteria
- [x] pretrain run を作れて、finetune が動く
- [x] upstream/downstream の追跡が meta でできる
