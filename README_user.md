# README_user

本ドキュメントは、LJ予測パイプラインにおける **前処理・特徴量化・学習モデル・推論** の機能と設定ポイントをまとめたユーザー向けの説明です。設定は Hydra の YAML で組み立てます（`configs/` 配下）。

---

## 1. 前処理（データセット作成）

### 1.1 機能一覧（build_dataset）

| 機能 | 内容 | 主な設定キー | 実装/参照 |
| --- | --- | --- | --- |
| データロード | 生CSVとSDFを読み込み | `paths.raw_csv`, `paths.sdf_dir`, `columns.*` | `src/data/adapters/lj.py` |
| LJパラメータ算出 | Tc/Pc/Tbから `epsilon/k`, `sigma` を計算 | `lj.*` | `src/common/lj.py` |
| 元素解析 | 分子式から元素集合・元素数を作成 | `columns.formula` | `src/common/chemistry.py` |
| 原子数算出 | SDFから `n_atoms`, `n_heavy_atoms` を算出 | `paths.sdf_dir` | `src/data/adapters/lj.py` |
| ターゲット登録 | target registry に従って列名を解決 | `task` + `target_registry` | `src/targets/registry.py` |
| 無効行除外 | LJ計算が欠損/範囲外の行を除外 | `filter_invalid_lj`, `lj.valid_range` | `src/data/adapters/lj.py` |
| セレクタ/サンプラ | 元素制約・頻度・サイズ・分布平坦化・多様性抽出 | `selectors[]` | `src/common/dataset_selectors.py` |
| Split | train/val/test を作成 | `split.*` | `src/common/splitters.py` |
| 出力 | processed CSV と index を保存 | `paths.out_csv`, `paths.out_indices_dir` | `src/data/adapters/lj.py` |

### 1.2 データセットアダプタ（`dataset.name`）

| dataset.name | 概要 | 主に使う config |
| --- | --- | --- |
| `lj` / `lj_quick` / `lj_fixture` | Tc/Pc/Tb から LJ を計算する本プロジェクト用 | `configs/dataset/default.yaml`, `configs/dataset/quick.yaml` |
| `qm9` / `qm9_quick` | QM9 分子データ | `configs/dataset/qm9.yaml`, `configs/dataset/qm9_quick.yaml` |
| `md17` | MD17 分子動力学 | `configs/dataset/md17.yaml` |
| `moleculenet` | MoleculeNet（ESOL/Freesolv/Lipo等） | `configs/dataset/esol.yaml`, `configs/dataset/freesolv.yaml`, `configs/dataset/lipo.yaml` |
| `pcqm4mv2` | PCQM4Mv2（OGB-LSC） | `configs/dataset/pcqm4mv2.yaml` |

### 1.3 セレクタ/サンプラ一覧（`selectors`）

| name | 目的 | 主な params | 補足 |
| --- | --- | --- | --- |
| `element_whitelist` | 許可元素のみ残す | `allowed` | 元素分布の安定化に有効 |
| `element_blacklist` | 禁止元素を除外 | `banned` | 逆フィルタ |
| `min_element_frequency` | 希少元素の行を除外 | `min_count` | 元素頻度に依存 |
| `max_size` | 分子サイズ上限 | `max_heavy_atoms` | `n_heavy_atoms` を利用 |
| `target_range` | 目的変数の範囲制限 | `target_col`, `valid_range` | 外れ値除去 |
| `target_stratified` | 分布平坦化 | `target_col`, `n_bins`, `samples_per_bin` | ダウンサンプル |
| `diversity_farthest_point` | 多様性最大化サンプリング | `n_samples`, `radius`, `n_bits` | Tanimoto を使用 |
| `butina_cluster` | クラスタ単位でサンプル | `dist_thresh`, `max_per_cluster` | Butina クラスタリング |

### 1.4 Split 方法（`split.method`）

| method | 概要 | 追加で必要なキー |
| --- | --- | --- |
| `random` | ランダム分割 | なし |
| `scaffold` | Murcko scaffold でリーク防止 | `paths.sdf_dir` / `columns.cas` |
| `group` | 同一グループを同じsplitへ | `split.group_key`（columns key） |
| `time` | 時系列順に分割 | `split.time_key`, `split.ascending` |

### 1.5 LJ推算式（`lj.*_method`）

| 対象 | method | 内容 |
| --- | --- | --- |
| epsilon/k | `bird_critical` | `0.77 * Tc` |
| epsilon/k | `bird_boiling` | `1.15 * Tb` |
| epsilon/k | `flynn` | `1.77 * Tc^(5/6)` |
| epsilon/k | `tee_gotoh_steward_1` | `0.774 * Tc` |
| sigma | `bird_critical` | `2.44 * (Tc/Pc_atm)^(1/3)` |

---

## 2. 特徴量化

### 2.1 Fingerprint / 記述子（Tabular）

| 種類 | 設定キー | 説明 | 参照 |
| --- | --- | --- | --- |
| Morgan FP | `featurizer.fingerprint: morgan` | 半径/bit数/カウント指定 | `src/fp/featurizer_fp.py` |
| MACCS FP | `featurizer.fingerprint: maccs` | 166 bitのMACCSキー | `src/fp/featurizer_fp.py` |
| RDKit FP | `featurizer.fingerprint: rdkit` | RDKit標準FP | `src/fp/featurizer_fp.py` |
| RDKit記述子 | `featurizer.add_descriptors` | 分子量/TPSA/LogP等 | `src/common/descriptors.py` |
| 欠損補完 | `preprocess.impute_nan` | `mean` など | `src/common/feature_pipeline.py` |
| 標準化 | `preprocess.standardize` | trueで標準化 | `src/common/feature_pipeline.py` |

**対応記述子**: `MolWt`, `HeavyAtomCount`, `TPSA`, `MolLogP`, `HBD`, `HBA`, `NumRotatableBonds`, `RingCount`, `AromaticRings`（`src/common/descriptors.py`）

### 2.2 GNNグラフ特徴（Graph）

| 機能 | 設定キー | 説明 | 参照 |
| --- | --- | --- | --- |
| ノード特徴 | `featurizer.node_features` | `atomic_num`, `degree` など | `src/gnn/featurizer_graph.py` |
| エッジ特徴 | `featurizer.edge_features` | `bond_type`, `distance`, `gaussian` | `src/gnn/featurizer_graph.py` |
| 半径グラフ | `featurizer.edge_mode: radius` | 3D距離で近傍構築 | `src/gnn/featurizer_graph.py` |
| 角度特徴 | `featurizer.angle_features` | DimeNet++用 `angle` | `src/gnn/featurizer_graph.py` |
| グローバル記述子 | `featurizer.add_global_descriptors` | 2D記述子の追加 | `src/gnn/featurizer_graph.py` |
| 3D座標 | `featurizer.use_3d_pos` | 3D必須モデルで必要 | `src/gnn/featurizer_graph.py` |
| コンフォーマー生成 | `conformer.*` / `featurizer.conformer.*` | ETKDG + UFF | `src/common/conformer.py` |

### 2.3 SMILESトークナイズ

| 種類 | 設定キー | 説明 | 参照 |
| --- | --- | --- | --- |
| ChemBERTa tokenizer | `featurizer.pretrained_name` | `seyonec/ChemBERTa-zinc-base-v1` | `src/smiles/tokenizer.py` |
| 文字数制限 | `featurizer.max_length` | 128/256 など | `src/smiles/tokenizer.py` |
| ローカル利用 | `featurizer.local_files_only` | trueでDL禁止 | `src/smiles/tokenizer.py` |

### 2.4 キャッシュ

| キャッシュ | パス | 説明 |
| --- | --- | --- |
| FP特徴量 | `data.cache_dir` | Fingerprint特徴量の再利用 | `src/fp/feature_utils.py` |
| Conformer | `runs/train/gnn/<exp>/artifacts/conformer_cache.json` | 3D座標の再利用 | `src/common/conformer.py` |

---

## 3. 学習モデル

### 3.1 モデル一覧（`model.name`）

| Family | model.name | 入力要求 | マルチターゲット | 備考 |
| --- | --- | --- | --- | --- |
| FP | `rf`, `lightgbm`, `catboost`, `gpr` | Tabular | 単一のみ | scikit-learn / LightGBM / CatBoost が必要 |
| GNN | `gcn` | nodeのみ | 対応 | 2D/3Dどちらも可 |
| GNN | `gin` | node（edge_attr任意） | 対応 | edge_attrがあるとGINEとして動作 |
| GNN | `mpnn` | edge_attr 必須 | 対応 | `edge_features` が必須 |
| GNN | `schnet` | pos + edge_attr | 単一のみ | 3D必須（距離特徴を利用） |
| GNN | `dimenetpp` | pos + angles | 単一のみ | 3D必須（角度） |
| GNN | `painn` | pos + edge_attr | 単一のみ | TorchMD-Net backend が必要 |
| GNN | `egnn` | pos | 対応 | 3D必須 | 
| GNN | `graphormer`, `graph_transformer` | nodeのみ | 対応 | 2DグラフTransformer |
| SMILES | `chemberta` | tokens | 対応 | Transformers が必要 |

**入力要求の補足**: `pos` は 3D 座標、`edge_attr` はエッジ特徴、`angles` は角度特徴、`tokens` は SMILES トークン列です（`src/models/registry.py`）。

### 3.2 学習設定（`train.*`）

| Backend | 主なキー | 意味 |
| --- | --- | --- |
| FP | `train.seed` | 乱数seed（分割・モデル再現性） |
| GNN | `epochs`, `batch_size`, `lr`, `weight_decay` | 学習の基本ハイパラ |
| GNN | `loss` | `mse` / `huber` など | 
| GNN | `early_stopping.patience` | 早期終了の待ち回数 |
| GNN | `device` | `auto` / `cpu` / `cuda` / `mps` |
| GNN | `num_workers`, `pin_memory` | DataLoader設定 |
| SMILES | `epochs`, `batch_size`, `lr`, `weight_decay` | Transformer学習パラメータ |
| SMILES | `freeze_encoder_epochs` | Encoder凍結期間（線形プローブ用途） |
| SMILES | `device`, `progress_bar`, `log_interval_sec` | 実行制御 |

### 3.3 事前学習/転移（`pretrain.*`）

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `ckpt_path` | 事前学習チェックポイント | `runs/train/.../model/model.ckpt` 等へのパス |
| `mode` | `finetune` / `linear_probe` / `partial_freeze` | 凍結戦略の切替 |
| `strict` | strict load の可否 | true でキー不一致を許さない |
| `freeze_patterns` | 凍結対象の正規表現 | `"encoder.*"` など |
| `trainable_patterns` | 学習対象の正規表現 | `"head.*"` など |

---

## 4. 推論

### 4.1 入力モード（`input.mode`）

| Backend | mode | クエリ | 動作 |
| --- | --- | --- | --- |
| FP | `formula` | 分子式 | `dataset_csv` からCASを解決 |
| FP | `cas` | CAS番号 | SDFから直接特徴量化 |
| GNN | `formula` / `cas` | 分子式 or CAS | SDF + conformer cache を使用 |
| SMILES | `smiles` | SMILES文字列 | そのままトークナイズ |
| SMILES | `cas` / `formula` | CAS/分子式 | `dataset_csv` からSMILES解決 |

### 4.2 不確かさ推定（`uncertainty.*`）

| method | 対応バックエンド | 説明 |
| --- | --- | --- |
| `null` | 全て | 不確かさ推定なし |
| `ensemble` | FP/GNN/SMILES | 複数モデルの分散を `y_std` として出力 |
| `mc_dropout` | GNN/SMILES | 推論時dropoutを複数回実行（FPは非対応） |

### 4.3 適用範囲診断（AD）

| 指標 | 由来 | 設定キー |
| --- | --- | --- |
| `trust_score` | 近傍Tanimoto + 未観測元素 | `ad.tanimoto_warn_threshold`, `ad.top_k` |
| `max_tanimoto` | 学習データの最近傍類似度 | `ad.*` |

FP/GNN は `runs/train/.../artifacts/ad.pkl` を参照し、SMILESはADなしです（`src/common/ad.py`）。

### 4.4 出力物

| 成果物 | 生成場所 | 内容 |
| --- | --- | --- |
| `predictions.csv` | `runs/predict/<exp>/` | 予測値（`y_pred`, `y_std` など） |
| `prediction_<CAS>.json` | `runs/predict/<exp>/` | 予測・AD・設定メタ | 
| `predict.log` | `runs/predict/<exp>/` | 推論ログ |

---

## 5. YAMLテンプレート

テンプレートは `templates/` に配置しています。必要に応じて `configs/` にコピーして編集するか、そのまま `--config` に指定して実行してください。

### 5.1 テンプレート一覧

| テンプレート | 目的 | 使うスクリプト |
| --- | --- | --- |
| `templates/dataset_template.yaml` | dataset作成 | `scripts/build_dataset.py` |
| `templates/fp_train_template.yaml` | FP学習 | `scripts/train.py` |
| `templates/fp_predict_template.yaml` | FP推論 | `scripts/predict.py` |
| `templates/gnn_train_template.yaml` | GNN学習 | `scripts/train.py` |
| `templates/gnn_predict_template.yaml` | GNN推論 | `scripts/predict.py` |
| `templates/smiles_train_template.yaml` | SMILES学習 | `scripts/train.py` |
| `templates/smiles_predict_template.yaml` | SMILES推論 | `scripts/predict.py` |

### 5.2 変数の意味と設定方法

#### `templates/dataset_template.yaml`

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `defaults.dataset` | datasetグループの選択 | `configs/dataset/*.yaml` のファイル名を指定 |
| `defaults.task` | ターゲット定義 | `configs/task/*.yaml` を指定（例: `lj_epsilon`） |
| `paths.raw_csv` | 生CSVのパス | ローカルパスを指定 |
| `paths.sdf_dir` | SDFディレクトリ | CAS一致のSDFを配置 |
| `columns.*` | 列名マッピング | CSVの列名に合わせる |
| `lj.epsilon_method` | epsilon推算式 | `bird_critical` 等を指定 |
| `lj.sigma_method` | sigma推算式 | `bird_critical` 等を指定 |
| `filter_invalid_lj` | 範囲外の除外 | true/false |
| `selectors[]` | フィルタ/サンプラ | `name` と `params` を追加 |
| `split.method` | split方式 | `random` / `scaffold` / `group` / `time` |
| `split.group_key` | group split のキー | `columns` で定義したキー名 |
| `split.time_key` | time split のキー | 時系列列のキー名 |
| `data.*` | 学習/推論で参照するパス | 作成後のCSV/indices/SDFを指定 |

#### `templates/fp_train_template.yaml`

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `defaults.features` | FP特徴量 | `configs/features/fp_*.yaml` を指定 |
| `defaults.model` | FPモデル | `configs/model/fp_*.yaml` を指定 |
| `preprocess.*` | 欠損補完/標準化 | `configs/preprocess/*.yaml` で切替 |
| `data.cache_dir` | FPキャッシュ | 再実行時に再利用したいパスを指定 |
| `ad.*` | AD閾値・近傍数 | `tanimoto_warn_threshold`, `top_k` |
| `output.run_dir` | 出力ルート | `runs/train/fp` など |
| `output.exp_name` | 実験名 | フォルダ名として使用 |

#### `templates/fp_predict_template.yaml`

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `model_artifact_dir` | 学習成果物のディレクトリ | `runs/train/fp/<exp>` を指定 |
| `model_artifact_dirs` | ensemble用の複数ディレクトリ | 複数指定で ensemble 推論 |
| `input.mode` | 入力モード | `formula` / `cas` |
| `uncertainty.method` | 不確かさ推定 | `ensemble` または `null` |
| `output.exp_name` | 推論出力名 | `runs/predict/<exp_name>` を作成 |

#### `templates/gnn_train_template.yaml`

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `defaults.features` | GNN特徴量 | `configs/features/gnn_*.yaml` を指定 |
| `defaults.model` | GNNモデル | `configs/model/*.yaml` を指定 |
| `train.*` | 学習ハイパラ | `epochs`, `batch_size`, `lr` など |
| `conformer.*` | 3D生成設定 | `enabled`, `method`, `forcefield` 等 |
| `pretrain.*` | 事前学習 | `ckpt_path` を指定する場合のみ有効 |
| `output.run_dir` | 出力ルート | `runs/train/gnn` など |

#### `templates/gnn_predict_template.yaml`

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `model_artifact_dir` | 学習成果物 | `runs/train/gnn/<exp>` を指定 |
| `input.mode` | 入力モード | `formula` / `cas` |
| `uncertainty.method` | 不確かさ推定 | `ensemble` / `mc_dropout` / `null` |
| `output.exp_name` | 推論出力名 | `runs/predict/<exp_name>` を作成 |

#### `templates/smiles_train_template.yaml`

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `defaults.dataset` | SMILES用データセット | 例: `qm9` |
| `defaults.features` | tokenizer設定 | `configs/features/smiles_*.yaml` |
| `defaults.model` | SMILESモデル | `configs/model/chemberta*.yaml` |
| `train.freeze_encoder_epochs` | Encoder凍結期間 | 0で無効、>0で線形プローブ |
| `data.smiles_col` | SMILES列 | データセットの列名に合わせる |

#### `templates/smiles_predict_template.yaml`

| キー | 意味 | 設定方法 |
| --- | --- | --- |
| `model_artifact_dir` | 学習成果物 | `runs/train/smiles/<exp>` を指定 |
| `input.mode` | 入力モード | `smiles` / `cas` / `formula` |
| `uncertainty.method` | 不確かさ推定 | `ensemble` / `mc_dropout` / `null` |
| `output.exp_name` | 推論出力名 | `runs/predict/<exp_name>` を作成 |

---

## 6. 実行例（テンプレート使用）

```bash
python3 scripts/build_dataset.py --config templates/dataset_template.yaml
python3 scripts/train.py --config templates/fp_train_template.yaml
python3 scripts/predict.py --config templates/fp_predict_template.yaml --query C6H6
```

Hydraの上書き例:

```bash
python3 scripts/train.py --config templates/gnn_train_template.yaml \
  model=gnn_mpnn train.epochs=50 train.batch_size=64
```
