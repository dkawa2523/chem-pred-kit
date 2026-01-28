# 化学構造ML学習基盤 設計案 v2（Hydra/CLIベース）


承知しました。リポジトリのコード内容を深く読み取り、以下2点を徹底的に調査・整理します：

承知しました。リポジトリのコード内容を深く読み取り、以下2点を徹底的に調査・整理します：

1. 【物性情報の特徴量候補一覧】

   * 既に使用されているかどうかを明示
   * 各物性情報の概要、SDFからの取得可否、その他取得方法、有用性、特に有用なケース、活用事例、今回の対象への優先度をテーブル形式で提示

2. 【学習アーキテクチャおよび事前学習系手法一覧】

   * 現在のリポジトリで使用されているものを明示
   * 各手法の概要、必要なデータ種・規模、有用性、特に有効なケース、ライブラリ、活用事例、今回の対象への優先度、計算リソース要件などを含めた比較テーブルを提示

完了次第、ご報告いたします。



本設計は、現行リポジトリ（LJ予測：FP/GCN/MPNN）を土台にして、将来 **物性追加・データ形式追加・特徴量追加・モデル追加・学習/推論/評価/可視化/サービング** まで一貫して拡張できる「化学系モデル学習基盤」に作り替えるための提案です。

---

## 1. 設計ゴール

### ユーザー視点（研究者・利用者）

* **Hydraで設定を切り替えるだけ**で

  * データ形式（SDF/SMILES/CSV…）
  * 特徴量（FP/記述子/3D/埋め込み…）
  * モデル（LGBM/GNN/3D-GNN/Transformer/事前学習…）
  * タスク（回帰/分類/マルチタスク…）
    を変更できる
* 1コマンドで **train / evaluate / predict / report** が実行できる
* 実験結果が `runs/` にまとまり **再現可能**（config + seed + データ版 + コード版が残る）

### 開発者視点（拡張・保守）

* 追加要素は **1ファイル追加 + registry登録 + config追加** で済む（既存コードを壊さない）
* “仕様がぶれない”＝ **I/O契約（データ・特徴量・モデル入出力）を固定**し、内部だけ差し替え可能
* 学習/推論の前処理ズレ（training-serving skew）を原理的に防ぐ
* 大規模化・運用に備え、**キャッシュ / バージョニング / 監視 hooks** を標準装備

---

## 2. 全体アーキテクチャ（レイヤ）

**Layer 0: CLI / Hydra（入口）**

* `scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py`, `scripts/report.py`

**Layer 1: Orchestrator（共通手順）**

* `PipelineRunner`（train/eval/predictの共通骨格）
* Hydra configから各コンポーネントを組み立てて実行

**Layer 2: Component（差し替え可能部品）**

* DatasetLoader / Splitter / Preprocessor
* Featurizer / FeatureTransform（scaler等）
* Model / Trainer / Evaluator / Calibrator（任意）
* Reporter / Visualizer

**Layer 3: Artifact & Registry（運用資産）**

* データ版・特徴量版・モデル版・実験版の管理
* すべて `runs/<exp_id>/` に集約し、将来はMLflow等へ移行可能

---

## 3. 推奨ディレクトリ構成

```text
project_root/
├── configs/                       # Hydra設定（差し替え可能部品ごとに分割）
│   ├── config.yaml                # ルート（defaults で組み合わせ）
│   ├── dataset/
│   │   ├── lj_csv_sdf.yaml
│   │   ├── smiles_csv.yaml
│   │   └── ...
│   ├── split/
│   │   ├── random.yaml
│   │   ├── scaffold.yaml
│   │   ├── time.yaml
│   │   └── ...
│   ├── preprocess/
│   │   ├── none.yaml
│   │   ├── neutralize.yaml
│   │   ├── salt_remove.yaml
│   │   ├── conformer_etkdg.yaml
│   │   └── ...
│   ├── features/
│   │   ├── fp_morgan.yaml
│   │   ├── rdkit_descriptors.yaml
│   │   ├── global_descriptors.yaml
│   │   ├── pretrained_embedding.yaml
│   │   └── ...
│   ├── transform/
│   │   ├── none.yaml
│   │   ├── standardize.yaml
│   │   ├── robust.yaml
│   │   └── ...
│   ├── model/
│   │   ├── lgbm.yaml
│   │   ├── catboost.yaml
│   │   ├── gcn.yaml
│   │   ├── mpnn.yaml
│   │   ├── schnet.yaml
│   │   ├── graph_transformer.yaml
│   │   ├── pretrained_finetune.yaml
│   │   └── ...
│   ├── task/
│   │   ├── regression.yaml
│   │   ├── classification.yaml
│   │   ├── multitask.yaml
│   │   └── ...
│   ├── train/
│   │   ├── default.yaml
│   │   ├── fast.yaml
│   │   ├── hpo_optuna.yaml
│   │   └── ...
│   ├── eval/
│   │   ├── default.yaml
│   │   └── ...
│   └── serving/
│       ├── batch.yaml
│       ├── api.yaml
│       └── ...
│
├── src/
│   ├── core/
│   │   ├── pipeline.py            # train/eval/predict の統一手順
│   │   ├── artifact.py            # Artifact I/O（保存/読込）
│   │   ├── io_contracts.py        # 入出力仕様（DataContract）
│   │   └── exceptions.py
│   ├── data/
│   │   ├── loaders/               # dataset loader plugins
│   │   ├── splitters/             # split plugins
│   │   ├── schemas.py             # データスキーマ（pydantic等）
│   │   └── cache.py               # キャッシュ管理
│   ├── preprocess/
│   │   ├── base.py
│   │   ├── neutralize.py
│   │   ├── salt_remove.py
│   │   └── conformer.py
│   ├── features/
│   │   ├── base.py
│   │   ├── fp.py
│   │   ├── rdkit_desc.py
│   │   ├── pretrained_embed.py
│   │   └── transforms.py          # scaler, imputer など
│   ├── models/
│   │   ├── base.py
│   │   ├── classical/             # LGBM, CatBoost 等
│   │   ├── gnn/                   # GCN, MPNN, GIN, GAT …
│   │   ├── threed/                # SchNet, DimeNet …
│   │   └── pretrained/            # 事前学習モデル利用
│   ├── tasks/
│   │   ├── base.py
│   │   ├── regression.py
│   │   ├── classification.py
│   │   └── multitask.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── calibration.py         # optional
│   ├── viz/
│   │   ├── plots.py
│   │   └── report.py
│   ├── registry.py                # registry（全プラグイン共通）
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── env.py
│
├── scripts/
│   ├── train.py                    # hydra entry
│   ├── evaluate.py
│   ├── predict.py
│   └── report.py
│
├── runs/                           # 実験成果物（Hydra run dir）
├── data/                           # raw/processed (DVC対象)
├── tests/                          # unit/integration
└── pyproject.toml / requirements.txt
```

---

## 4. “仕様がぶれない”ための 핵（契約設計）

### 4.1 DataContract（入力データの共通仕様）

* **最低限これだけは全タスクで揃える**

  * `id`（CAS/SMILES/内部ID など一意）
  * `mol`（RDKit Mol もしくは SMILES）
  * `y`（ターゲット：1次元 or 多次元）
  * `meta`（任意：分子式、データソース、温度条件など）

**ポイント**

* 各DatasetLoaderは最終的に `List[Sample]`（or DataFrame + Mol列）を返す
* どのデータ形式でも **同じ中間表現** に揃える → 下流はデータ形式を気にしない

### 4.2 FeatureContract（特徴量の共通仕様）

* 特徴量は `FeatureBatch` として統一

  * `x_tabular`（numpy/torch：固定長ベクトル）
  * `x_graph`（PyG Data/Batch：グラフ）
  * `x_3d`（座標・距離行列など）
  * `x_text`（SMILES tokenなど）
* **複数同時に持てる**（例：GNN + global descriptors）

### 4.3 ModelContract（モデルの入出力仕様）

* `model.forward(batch: FeatureBatch) -> y_pred`
* タスク側が `loss_fn(y_pred, y_true)` と `metrics(y_pred, y_true)` を管理

---

## 5. Registry（プラグイン）設計

### 5.1 登録対象

* dataset loader / splitter / preprocessor
* featurizer / transform
* model
* task
* evaluator / metric

### 5.2 追加手順を固定化（開発者導線）

例：新しいFeaturizerを追加

1. `src/features/my_new_feat.py` を追加（`BaseFeaturizer`継承）
2. `@register("features", "my_new_feat")` で登録
3. `configs/features/my_new_feat.yaml` を追加
4. `configs/config.yaml` のdefaultsから参照可能に

---

## 6. Hydra設定設計（例）

### 6.1 ルート `configs/config.yaml`

```yaml
defaults:
  - dataset: lj_csv_sdf
  - split: scaffold
  - preprocess: [neutralize, salt_remove]
  - features: [fp_morgan, rdkit_descriptors]
  - transform: standardize
  - model: lgbm
  - task: regression
  - train: default
  - eval: default
  - _self_

seed: 42
run_name: ${now:%Y%m%d_%H%M%S}_${model.name}_${dataset.name}
output_dir: runs/${run_name}
```

### 6.2 実行コマンド例

```bash
# ベースライン学習
python scripts/train.py

# モデル切替（GNN）
python scripts/train.py model=mpnn features=[global_descriptors] train=default

# 3Dモデル
python scripts/train.py model=schnet preprocess=[conformer_etkdg] features=[3d_coords]

# 推論
python scripts/predict.py +ckpt_path=runs/20251227_.../model.ckpt +input=inputs.csv

# HPO（optuna設定をtrainに切替）
python scripts/train.py train=hpo_optuna -m
```

---

## 7. Artifact設計（運用上の事故を防ぐ）

各実験 `runs/<exp_id>/` に **必ず** 以下を保存：

```text
runs/<exp_id>/
├── config_resolved.yaml        # Hydraが解決した最終設定（超重要）
├── dataset_fingerprint.json    # データ版（hash、件数、split種別など）
├── feature_fingerprint.json    # 特徴量版（使用featurizer一覧、次元、統計）
├── model.ckpt                  # モデル本体 + 前処理(transform)状態
├── metrics.json                # 指標（train/val/test）
├── predictions.csv             # 必要なら（test予測）
├── logs.txt                    # ログ
└── plots/                      # 学習曲線、残差、重要特徴など
```

**重要な設計ポイント**

* `transform`（imputer/scaler/label transform）を **モデルと一緒に保存**
  → 推論時に同じ変換を自動で再適用し、training-serving skewを防ぐ
* `dataset_fingerprint` にデータのハッシュやsplit seedを保存
  → データが変わったのに同じモデルとして扱ってしまう事故を防止

---

## 8. 想定課題への対応（設計で潰す）

| 想定課題      | 典型症状               | 設計での対策                                                 |
| --------- | ------------------ | ------------------------------------------------------ |
| スキーマドリフト  | ある日突然パイプラインが壊れる    | `schemas.py`でスキーマ検証、Loaderで吸収、`dataset_fingerprint`に記録 |
| データ版不明    | 「どのデータで学習した？」が追えない | データhash + split情報をArtifactに固定保存（+ DVC推奨）               |
| 特徴量の不整合   | 学習と推論で変換が違い精度崩壊    | `FeatureTransform`を保存/ロード、Predictorが必ず同一パイプラインを通す      |
| 新特徴量追加が地獄 | あちこち修正が波及          | FeatureContract + registryで追加箇所を固定化                    |
| 新モデル追加が地獄 | trainerやIOまで改修が必要  | ModelContract + Task分離 + registryで追加箇所を固定化             |
| Hydraが複雑化 | 何が有効か分からない         | config group整理、命名規約、`config_resolved.yaml`必須保存         |
| 大規模化で遅い   | 前処理/特徴量生成がボトルネック   | `data/processed`にキャッシュ、fingerprint/descriptorを永続化      |
| 実験比較が難しい  | どれが良いモデルか分からない     | `metrics.json`標準化、`report.py`で集計レポート自動生成               |
| デプロイしづらい  | CLIは動くがAPIで動かない    | coreを関数/クラス化（Predictor）、CLIは薄くする                       |
| 更新が怖い     | 新モデルで劣化→戻せない       | モデル版管理、旧版を残す、評価ゲート（しきい値で失敗させる）                         |

---

## 9. 学習・推論・評価・可視化・サービングの標準フロー

### 9.1 Train（学習）

1. DatasetLoader → Sample list
2. Splitter → train/val/test index
3. Preprocess pipeline（Mol整形/3D生成）
4. Featurizer pipeline → FeatureBatch
5. Transform fit（trainのみ）→ transform保存
6. Model/Task/Trainerで学習
7. Artifact保存（config/transform/model/metrics）

### 9.2 Predict（推論）

1. ckptロード → model + transform + featurizer設定
2. 入力（SMILES/CSV/SDF）を同じPipelineで処理
3. 予測
4. CSV/JSON出力（標準フォーマット）

### 9.3 Evaluate（評価）

* evaluatorが `task.metrics` を呼ぶだけで良い（モデル差を持ち込まない）
* 追加メトリクスは `metrics.py` に追加し registry登録

### 9.4 Report（可視化/レポート）

* `runs/*/metrics.json` を集約し

  * 学習曲線
  * 残差プロット
  * ターゲット分布・予測分布
  * 重要特徴（可能なモデルのみ）
    を `runs/_reports/` に出力

### 9.5 Serving（API化）

* `src/core/predictor.py` に `Predictor.load(ckpt)` を提供
* FastAPI等は別レイヤ（`serving/`）で `Predictor` を呼ぶだけ
  → 学習基盤とサービング基盤の疎結合

---

## 10. 実装優先順位（最短で基盤化する順）

1. **Artifact + config_resolved保存（再現性の核）**
2. **DataContract / FeatureContract（仕様固定）**
3. **Train/Predict/Evaluateの統一PipelineRunner**
4. **Registry導入（モデル/特徴量/データの追加容易化）**
5. **Transform保存（training-serving skew根絶）**
6. **キャッシュ（processed features）**
7. **Report集計（実験比較の標準化）**
8. **Serving（PredictorのAPI化）**
9. **HPO/分散/監視（必要に応じて拡張）**

---

## 11. 追加で入れると強い運用・保守の工夫

* **テスト**

  * `tests/unit/`：各Featurizerが返すshape、欠損処理、Mol処理
  * `tests/integration/`：train→predictが同じ結果になる（最重要）
* **CI（GitHub Actions）**

  * lint（ruff/black）
  * unit test
  * minimal training smoke test（小データで1epoch）
* **依存の分割**

  * `extras_require`（`[gnn]`, `[3d]`, `[dev]`）で軽量利用可能に
* **データ管理**

  * DVC導入（`data/raw` と `data/processed` を版管理）
* **評価ゲート**

  * `eval.min_r2` などを設定し、基準未達ならジョブを失敗扱いに（運用事故防止）

---

