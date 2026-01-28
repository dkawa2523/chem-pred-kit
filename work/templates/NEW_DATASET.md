# NEW_DATASET テンプレート

# New Dataset/Property: <名前>

## 目的
- 追加する物性（目的変数）

## 入力
- CSV列（target列名）
- SDFの扱い

## 実装スコープ
- dataset config 追加
- task 定義（loss/metrics）
- 収集機能が必要なら data_collection も

## 受け入れ条件
- [ ] `task=<name>` で学習できる
- [ ] 評価指標が出る
