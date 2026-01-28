# ChemKit Next 拡張概要（090完了後の次フェーズ）

このフォルダは `test_chemkit` ブランチで **Task 090 まで完了した後**に、
化学ML基盤を **ベンチマーク（QM9/MD17/MoleculeNet 等）・3D GNN・マルチタスク・事前学習**へ拡張するための
「不変に近い仕様（Spec）」をまとめます。

## 目的
- **データセット追加**が「Adapter追加＋Config追加」で完結する
- **モデル追加**が「Model実装＋Registry登録＋Config追加」で完結する
- train/eval/predict/visualize を **同じ Artifact 契約**で比較可能にする
- optional dependency（ライブラリ差）を許容しつつ、ベース環境を壊さない
- 将来的な ClearML / 実験管理にも耐える（run単位で入力/出力が追跡できる）

## スコープ
- QM9/MD17/MoleculeNet/PCQM4Mv2 等の対応（段階的）
- 3Dモデル（SchNet/DimeNet++/PaiNN/EGNN 等）を段階追加
- SMILES Transformer（ChemBERTa 等）の導入
- マルチタスク（複数物性の同時学習）
- 事前学習（教師あり/自己教師あり）→ 転移学習（fine-tune/linear-probe）
- 可視化・比較・レポート（leaderboard 拡張）

## 非スコープ（このSpec群だけでは実装しない）
- ClearML そのものの実装（準備設計はする）
- 研究用途以外の本番Serving（API）実装（最低限のpredictは扱う）

## 既存（090まで）の前提
- Process単位のCLIが成立している（train/evaluate/predict/visualize等）
- Artifact契約（config/meta/metrics/predictions/model）と contract test が存在する
- split戦略（random/scaffold/group等）と漏洩検知の基礎がある

> このSpecは **docs/00_INVARIANTS.md** と矛盾しないことが前提です。
