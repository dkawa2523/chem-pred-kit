# Task 035 (P0): split戦略の追加（random/scaffold/group、seed固定、漏洩防止）

## 目的（Why）
- R²の比較が意味を持つように、splitを戦略的に選べるようにする。
- splitが変わるとスコアが大きく変動するため、**同一splitで比較**できる仕組みが必須。

## 背景（Context）
- random splitだけだと簡単に“高スコア”が出る（漏洩/類似分子が跨る）。
- 分子タスクでは scaffold split が標準的な比較になりやすい。
- 将来の比較（multirun, ClearML）でも split の追跡が必須（dataset_hash + split.json）。

## スコープ（Scope）
### In scope
- `build_dataset`（またはsplit生成部分）に split戦略を追加
  - `random`（seed固定）
  - `scaffold`（Murcko scaffoldでグルーピングし、グループ単位で割当）
  - `group`（指定列：例 cas / inchikey / formula 等、同一グループは同一split）
- split成果物の保存（契約）
  - `split.json` などとして artifact に保存（docs/04準拠）
- splitのバリデーション
  - split間重複がない
  - scaffold split では同一scaffoldが跨らない（少なくとも検査できる）

### Out of scope（今回はやらない）
- Nested CV、完全なk-fold運用（これはP1でもOK）
  - ただし “将来追加できる拡張ポイント” は設計しておく

## 影響（Contract Impact）
- dataset artifactの内容が増える（split保存）
- 既存のsplit生成がある場合、互換性を壊さずに拡張する（旧キーを残す/変換する）

## 実装計画（Plan）
1) `src/data/splitting.py`（案）を追加し、splitメソッドを関数として実装
   - `make_split_random(...)`
   - `make_split_scaffold(...)`（RDKit: Murcko scaffold）
   - `make_split_group(...)`
2) `configs/dataset/*.yaml` に以下のキーを追加（例）
   - `dataset.split.method: random|scaffold|group`
   - `dataset.split.seed: 0`
   - `dataset.split.fractions: [0.8, 0.1, 0.1]`
   - `dataset.split.group_key: "cas"`（groupの場合）
3) `build_dataset` Process に組込み、splitを生成して artifact に保存
4) テスト追加（重要）
   - 小さな分子集合で scaffold split を作り、同一scaffoldが跨らないことを検査
   - seed固定で split が再現されることを検査

## 受け入れ条件（Acceptance Criteria）
- [x] random/scaffold/group を config で切替できる
- [x] seed固定で split が再現される
- [x] split成果物が artifact として保存される（docs/04準拠）
- [x] scaffold split の漏洩検査ができる（テストで担保）
- [x] pytestに最低1つ追加テスト

## 検証手順（How to Verify）
- 例：
  - `python scripts/build_dataset.py dataset.split.method=scaffold dataset.split.seed=0`
  - `python scripts/build_dataset.py dataset.split.method=group dataset.split.group_key=cas`
  - `pytest -q`

## メモ
- このタスク完了後、train/evaluate は “dataset artifactのsplitを必ず使う” 方針に寄せる（比較可能性）
