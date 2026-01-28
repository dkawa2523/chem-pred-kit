# Task 075 (P0): 結果集計Process（leaderboard/比較レポート生成）

## 目的（Why）
- モデル/特徴量/タスクの比較を「人力でrunフォルダを見に行く」状態から脱却する。
- multirunで大量実験したとき、**どれが良いか即分かる**ようにする。
- 将来ClearMLに移行しても、同じ “比較レポート” の発想で繋げられる。

## 背景（Context）
- 比較の正しさには、(a)同一split、(b)同一指標定義、(c)metaが揃っていることが必要。
- `metrics.json` / `meta.json` / `config.yaml` を契約化しているので、集計は自動化できる。

## スコープ（Scope）
### In scope
- **新Process `leaderboard`（または `aggregate_results`）** を追加（1 script = 1 process）
  - `scripts/leaderboard.py`（例）
- 入力：`runs/` 配下（または指定root）をスキャンし、以下が揃うrunを集計
  - `meta.json`
  - `metrics.json`
  - （任意）`config.yaml`
- 出力（artifact）
  - `leaderboard.csv`
  - `leaderboard.md`（上位N件、条件付き）
  - `plots/`（任意：metric vs timeなど）

### Out of scope（今回はやらない）
- ClearML SDKでのアップロード（設計準拠だけ）
- 完全なWeb UI

## 影響（Contract Impact）
- Process追加なので `docs/10_PROCESS_CATALOG.md` を更新
- 既存runを壊さない（read-onlyで集計する）

## 実装計画（Plan）
1) `docs/10_PROCESS_CATALOG.md` に leaderboard を追記（入力/出力）
2) `scripts/leaderboard.py` を追加
   - `leaderboard.root_dir`（デフォルト runs/）
   - `leaderboard.metric_key`（例 r2）
   - `leaderboard.sort_order`（desc）
   - `leaderboard.filters`（task/model/features、期間など）
3) `configs/process/leaderboard.yaml`（入口）と `configs/leaderboard/default.yaml` を追加
4) テスト追加
   - `tests/` でテンポラリrunディレクトリを作り、ダミーの meta/metrics を置いて集計できること

## 受け入れ条件（Acceptance Criteria）
- [x] `python scripts/leaderboard.py ...` が単独で実行できる（Hydra管理）
- [x] `leaderboard.csv` が生成される
- [x] 필터（task/model/featuresの少なくとも1つ）が動く
- [x] `docs/10_PROCESS_CATALOG.md` が更新されている
- [x] pytestに最低1つ追加テスト

## 検証手順（How to Verify）
- 例：
  - `python scripts/leaderboard.py leaderboard.root_dir=runs leaderboard.metric_key=r2`
  - `pytest -q`
- 実行済み：
  - `pytest -q tests/test_leaderboard.py`

## メモ
- これができると、HPOやアンサンブルの比較が一気に回しやすくなる
