# Repo Orientation (S00)

この文書は agentskills/S00_repo_orientation の出力です。

## 1) ルート構成マップ
- agentskills/: Codex向けスキル群（作業手順/ルーティング）
- codex/: セッション文脈・プロンプトテンプレ・運用チェック
- configs/: 実行設定（dataset/fp/gnn + quick）
- data/: raw/processed データ置き場（ローカル）
- docs/: 不変条件/契約/設計/ADR
- scripts/: CLIエントリポイント（dataset/train/evaluate/predict/visualize）
- src/: 実装本体（common/fp/gnn）
- tools/: 補助スクリプト（Codex向け）
- work/: タスク/バックログ/RFC/テンプレ
- runs/: 実行成果物の出力先（実行時に生成）
- requirements.txt: Python依存

## 2) scripts/ 実行動線（dataset → train → predict）

```
raw CSV + SDF
  -> scripts/build_dataset.py
     -> data/processed/dataset_with_lj.csv
     -> data/processed/indices/*.txt
     -> data/processed/dataset_config_snapshot.yaml

processed dataset + indices + SDF
  -> scripts/train.py (FP config)
     -> runs/train/fp/<exp>/{artifacts,plots,metrics_*.json,config_snapshot.yaml}
  -> scripts/train.py (GNN config)
     -> runs/train/gnn/<exp>/{artifacts,plots,metrics_*.json,config_snapshot.yaml}

model artifacts + SDF (+ dataset for formula resolve)
  -> scripts/predict.py
     -> runs/predict/<exp>/prediction_<CAS>.json
```

## 3) 主要スクリプトの入出力一覧

| Script | Config | Inputs | Outputs |
|---|---|---|---|
| scripts/build_dataset.py | configs/dataset*.yaml | data/raw/tc_pc_tb_pubchem.csv, data/raw/sdf_files | data/processed/dataset_with_lj.csv, data/processed/indices/*.txt, data/processed/dataset_config_snapshot.yaml, data/processed/build_dataset.log |
| scripts/train.py | configs/fp/train*.yaml | data/processed/dataset_with_lj.csv, data/processed/indices, data/raw/sdf_files | runs/train/fp/<exp>/artifacts/{model.pkl,imputer.pkl,scaler.pkl?,ad.pkl}, runs/train/fp/<exp>/plots/*.png, runs/train/fp/<exp>/metrics_{val,test}.json, runs/train/fp/<exp>/config_snapshot.yaml, runs/train/fp/<exp>/train.log |
| scripts/train.py | configs/gnn/train*.yaml | data/processed/dataset_with_lj.csv, data/processed/indices, data/raw/sdf_files | runs/train/gnn/<exp>/artifacts/{model_best.pt,graph_featurizer.pkl,ad.pkl?}, runs/train/gnn/<exp>/plots/*.png, runs/train/gnn/<exp>/metrics_{val,test}.json, runs/train/gnn/<exp>/config_snapshot.yaml, runs/train/gnn/<exp>/train.log |
| scripts/predict.py | configs/fp/predict*.yaml | runs/train/fp/<exp> (artifacts + config_snapshot), data/raw/sdf_files, data/processed/dataset_with_lj.csv | runs/predict/<exp>/prediction_<CAS>.json, runs/predict/<exp>/predict.log |
| scripts/predict.py | configs/gnn/predict.yaml | runs/train/gnn/<exp> (artifacts + config_snapshot), data/raw/sdf_files, data/processed/dataset_with_lj.csv | runs/predict/<exp>/prediction_<CAS>.json, runs/predict/<exp>/predict.log |

補足: Fingerprint の特徴量キャッシュは config で指定した場合に `data/processed/cache/fp/` 配下へ作成されます。

## 4) configs/ 現状と重複/混乱ポイント

現状の設定ファイル:
- configs/dataset.yaml
- configs/dataset_quick.yaml
- configs/fp/train.yaml
- configs/fp/train_quick.yaml
- configs/fp/predict.yaml
- configs/fp/predict_quick.yaml
- configs/gnn/train.yaml
- configs/gnn/train_quick.yaml
- configs/gnn/train_mpnn_quick.yaml
- configs/gnn/predict.yaml

重複/混乱ポイント（現状観測）:
- quick 系が複数ファイルに分散しており、差分が増えると同期漏れが発生しやすい
- `gnn/train_quick.yaml` と `gnn/train_mpnn_quick.yaml` が並存し、どちらが基準か読み取りにくい
- Hydra の group 構成が未整理で、dataset / model / predict の切替粒度が統一されていない

## 5) 次に整理すべき P0（候補3つ）
- 010: Hydra config group 再整理（重複の削減と命名統一）
- 015: Process 単位の CLI エントリポイント整理（1 Process = 1 CLI）
- 020: Artifact 契約の実装（docs/04 の実装化）
