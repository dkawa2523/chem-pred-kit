# Process Catalog（処理単位の一覧とI/O）

このドキュメントは **Process（処理単位）** のカタログです。  
「どのProcessが存在し、何を入力として、何を出力するか」を固定し、拡張してもブレないようにします。

---

## 1. Process一覧（推奨）
| Process | Script（推奨） | 主入力 | 主出力（artifact） | ClearML化の単位 |
|---|---|---|---|---|
| build_dataset | scripts/build_dataset.py | raw CSV/SDF | dataset index/split, dataset_hash | Dataset Task |
| audit_dataset | scripts/audit_dataset.py | raw CSV/SDF or processed dataset | audit_report.json, audit_report.md, plots | Audit Task |
| featurize | scripts/featurize.py | dataset artifact | features cache/manifest | Task |
| train | scripts/train.py | dataset(+features), task, model, train cfg | model.ckpt, metrics | Training Task |
| pretrain | scripts/pretrain.py | dataset(+features), self-supervised cfg | model.ckpt, metrics | Training Task |
| evaluate | scripts/evaluate.py | model.ckpt + dataset | metrics, predictions | Evaluation Task |
| predict | scripts/predict.py | model.ckpt + new input | predictions | Inference Task |
| visualize | scripts/visualize.py | predictions/metrics | plots | Reporting Task |
| leaderboard | scripts/leaderboard.py | runs/ (meta.json + metrics.json) | leaderboard.csv, leaderboard.md | Reporting Task |
| benchmark_suite | scripts/benchmark_suite.py | benchmark matrix YAML | benchmark_report.md, benchmark_plan.json, leaderboard outputs | Reporting Task |
| collect_data | scripts/collect_data.py | API cfg | raw CSV/SDF | Data Collection Task |

※ 現在のrepoに scripts が揃っていない場合は、まず “入口を分ける” ことを P0 とする。

---

## 2. Process共通I/O規約
### 2.1 入力は「上流artifact参照」を優先
- `train` は raw CSV/SDF を直接読むよりも、`build_dataset` の artifact を参照する（推奨）
- ただし研究用途で簡易実行が必要なら raw 入力も許す（configで切替）

### 2.2 出力は artifact 契約に従う
- `config.yaml`, `meta.json` は全Process必須
- `upstream_artifacts` に参照元runを残す

---

## 3. 拡張のやり方（Process追加/分割）
### 3.1 追加時にやること
- scripts に新しい entrypoint を追加（1 process = 1 script）
- configs/process に設定を追加（推奨）
- docs/10 に追記
- work/tasks にタスク化し、smoke test を追加

### 3.2 分割の目安
- “改良頻度が高い機能” と “安定した機能” が混ざって肥大化したら分割
  - 例：trainに可視化が混ざってきたら visualize を独立Processへ

---

## 4. ClearML移行のための準備（今やるべきこと）
- 全Processで `meta.json` を統一
- `dataset_hash` / `config_hash` を必ず記録
- 依存関係（上流artifact参照）を `upstream_artifacts` に残す
