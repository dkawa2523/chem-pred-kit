# Test Policy（テスト方針）

このドキュメントは「何十回の改修で壊れない」ためのテスト戦略です。

## 1. テストの種類
### 1.1 Unit
- 小さな関数/クラス（特徴量、メトリクスなど）
- 実行が速い（数秒以内）

### 1.2 Integration（smoke）
- CLI で学習/推論が最小データで完走する
- 成果物（artifact）が契約どおり出る

### 1.3 Contract（ゴールデン）
- `predictions.csv` の列や `meta.json` のキーが壊れていないか
- config validation が期待通り落ちるか（無効組合せの検出）

## 2. 最低ライン（PRの受け入れ条件）
- 新機能/修正には **最低1つ** テストを追加する
- `pytest -q` が通ること
- smoke test（軽量 config）が通ること

## 3. 推奨構成
- `tests/unit/`
- `tests/integration/`
- `tests/contract/`
