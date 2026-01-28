# S53: 3D GNN Models (SchNet/DimeNet++/PaiNN/EGNN)

## 目的
- 3Dモデルを optional dependency として追加しつつ、比較可能性を保つ

## 実装ガイド
- model registry に name→constructor を登録
- required inputs を宣言（pos, edge_attr, angles 等）
- importは lazy に（依存が無い場合は明確なエラー）
- config group を追加（model=schnet 等）

## テスト
- smoke: 10分子で train→eval が走る
- contract: artifacts が保存される（model/metrics/preds/meta）
