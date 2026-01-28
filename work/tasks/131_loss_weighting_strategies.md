# Task 131 (P1): Loss Weighting Strategies

## 目的
マルチタスクのターゲット間スケール差を緩和する。

## In scope
- configで `loss.weights` を指定可能に
- （optional）uncertainty weighting/GradNorm を段階導入
- レポートに weight を記録（meta/metrics）

## Acceptance Criteria
- [x] weight指定で学習が動く
- [x] タスク別metricが改善することを確認できる（少なくとも比較可能）
