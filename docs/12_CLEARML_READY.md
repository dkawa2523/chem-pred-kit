# ClearML-Ready Design（将来連携のための設計）

このドキュメントは「将来ClearMLに載せる」ための準備チェックリストです。  
**今は実装しない**が、設計・成果物はClearMLと相性が良い形にしておく。

---

## 1. ClearML Task 化の基本方針
- 1 Process = 1 ClearML Task（原則）
- Taskは “入力artifact” と “出力artifact” を持つ
- Taskのパラメータは Hydra config で一意に決まる

---

## 2. 今から守るべきこと（実装不要・成果物/設計で担保）
- すべての処理が non-interactive（対話入力なし）
- `config.yaml` が常に保存される
- `meta.json` に以下が常に入る：
  - `process_name`, `run_id`, `dataset_hash`, `config_hash`, `git_sha`
  - `upstream_artifacts`
  - `tags`

---

## 3. 将来導入する時の想定（参考）
- `clearml` config group を追加し、`enable: true/false` を切替
- enable時にだけ `clearml.Task.init(...)` を呼び、config を connect
- artifact upload は Process終了時にまとめて行う（model/metrics/predictions/plots）

※ ClearML固有コードは `src/integrations/clearml/` に隔離する（境界を守る）。
