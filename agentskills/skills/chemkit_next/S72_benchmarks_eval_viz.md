# S72: Benchmarks / Evaluation / Visualization

## 目的
- データセット慣習に沿った評価と比較レポートを自動生成する

## 推奨
- QM9: MAE中心、ターゲットごとに表
- MD17: energy/forces を分けて評価
- parity/residual/embedding を artifacts に保存

## 注意
- split は固定（保存済みを必ず使う）
- 失敗ケースは dataset audit に回す
