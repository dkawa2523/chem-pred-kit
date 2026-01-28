# Artifacts & Versioning（成果物・再現性）

このドキュメントは「各Processが独立して実行でき、比較・再現できる」ための成果物（artifact）契約です。  
将来ClearML Task化を見越し、**Process単位で入力/出力が追跡できる**形にします。

---

## 1. 出力ディレクトリ（推奨形）
Hydraの出力ディレクトリ（例：`runs/<process>/<date>/<time>_<experiment>/`）に保存する。

- `<process>` は `train`/`predict`/`evaluate` 等の処理単位
- 1 run = 1 Process 実行

---

## 2. 必須ファイル（全Process共通）
- `config.yaml`：最終 config（Hydra合成結果）
- `meta.json`：メタ情報（下記必須キー）
- `logs/`：ログ（形式は自由だが存在推奨）

---

## 3. Process別の必須成果物
### 3.1 build_dataset
- `dataset/`
  - `dataset_index.csv`（sample_id一覧、split含めてもよい）
  - `split.json`（train/val/test index）
- `dataset_hash.txt` または `meta.json` 内に `dataset_hash`

### 3.2 featurize
- `features/`
  - `features_manifest.json`（特徴量の説明・次元・バージョン）
  - キャッシュ（必要なら）
- 重要：学習/推論で同じfeaturesを使えること

### 3.3 train
- `model/`
  - `model.ckpt`
  - `preprocess.pkl`（必要なら：scaler/imputer等）
  - `featurizer_state.json`（必要なら：語彙/辞書/設定など）
- `metrics.json`（train/valの最終 or best）
- `plots/`（任意）

### 3.4 evaluate
- `metrics.json`（test等の評価）
- `predictions.csv`（比較可能な列）
- `plots/`（任意）

### 3.5 predict
- `predictions.csv`
- （任意）`uncertainty.csv`（不確実性がある場合）

### 3.6 visualize
- `plots/`（生成した図を全てここへ）

### 3.7 audit_dataset
- `audit/`
  - `audit_report.json`
  - `audit_report.md`
- `plots/`（分布/外れ値など）

---

## 4. meta.json の必須キー（ClearML-ready）
- `run_id`：一意ID
- `process_name`：例 `train`
- `created_at`
- `git_sha`（可能なら）
- `dataset_hash`
- `config_hash`
- `task_name`
- `model_name`
- `featureset_name`
- `upstream_artifacts`：上流runの参照（パスやrun_idのリスト）
- `tags`：任意（将来ClearMLのフィルタに使う）

---

## 5. predictions.csv の列（比較可能性のための契約）
- 必須：
  - `sample_id`
  - `y_pred`
- 推奨：
  - `y_true`（評価時）
  - `y_std`（不確実性がある場合）
  - `split`（train/val/test）
  - `model_name`, `model_version`
  - `dataset_hash`, `run_id`

列追加はOKだが、必須列を壊さない。

---

## 6. 集計（将来の比較評価）
将来、複数runの `metrics.json` を集計して leaderboard を作る想定。  
そのため `metrics.json` には少なくとも以下を含めるのを推奨：
- `r2`, `mae`, `rmse`（回帰）
- `accuracy`, `auc`（分類）
- `n_train`, `n_val`, `n_test`
- `seed`
- （マルチタスク）`by_split` 内に `by_target` を含める／または `by_target`（target -> split -> metrics）

（どの指標を必須にするかはタスクに依存するため、最終決定は ADR で行う）
