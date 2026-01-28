# 目的 / ゴール

このプロジェクトは、**臨界定数 `Tc/Pc`（必要なら `Tb`）から Lennard–Jones (LJ) パラメータ `epsilon/k`・`sigma` を推算**し、さらに **分子構造（SDF）を使ったMLでLJを予測**できるようにするための「設定駆動（YAML）」パイプラインです。

- データ作成：`Tc, Pc, Tb` → `lj_epsilon_over_k_K`, `lj_sigma_A` を列追加
- データ選別：疎な元素分布/サイズ/多様性を考慮したフィルタ・サンプリング
- 学習（Fingerprint系）：Morgan/MACCS/RDKit FP + 任意の記述子 → RF/CatBoost/LightGBM/GPR
- 学習（GNN系・任意）：SDF → 分子グラフ（PyTorch Geometric） → GCN/MPNN
- 推論：CAS or 分子式 → 予測 + Applicability Domain（外挿警告）

---

# 入力データ

## `data/raw/tc_pc_tb_pubchem.csv`

最低限以下の列が必要です（デフォルト）：

- `CAS`
- `MolecularFormula`
- `Tc [K]`
- `Pc [Pa]`
- `Tb [K]`

列名が違う場合は `configs/dataset.yaml` の `columns:` でマッピングできます。

## `data/raw/sdf_files/<CAS>.sdf`

- ファイル名は **CSVのCASと一致**させます（例：`71-43-2.sdf`）
- SDFが無い/壊れている行は、サイズ列や特徴量が欠損し、学習時に除外されます

---

# 処理フロー（何がどこで起きるか）

## 1) データセット作成：`scripts/build_dataset.py`

- 分子式パース：`src/common/chemistry.py`
  - `elements`（例：`C,H,O`）と `n_elements` を追加
- SDF読込：`src/common/io.py`
  - `n_atoms`, `n_heavy_atoms` を追加
- LJ推算：`src/common/lj.py`
  - `lj_epsilon_over_k_K`, `lj_sigma_A`（デフォルトは Bird/Stewart/Lightfoot）
- 選別パイプライン：`src/common/dataset_selectors.py`
  - `configs/dataset.yaml` の `selectors:` を上から順に適用
- Split：`src/common/splitters.py`
  - `random` / `scaffold`

出力：

- `data/processed/dataset_with_lj.csv`
- `data/processed/indices/{train,val,test}.txt`
- `data/processed/dataset_config_snapshot.yaml`

## 2) Fingerprint学習：`scripts/train.py`

- 特徴量：`src/fp/featurizer_fp.py` + `src/common/descriptors.py`
- 学習器：`src/fp/models.py`
- 成果物保存：`runs/train/fp/<exp_name>/...`
  - `artifacts/model.pkl`（モデル）
  - `artifacts/imputer.pkl`（欠損補完）
  - `artifacts/scaler.pkl`（標準化を有効にした場合）
  - `artifacts/ad.pkl`（AD用：訓練FP・元素集合）

## 3) Fingerprint推論：`scripts/predict.py`

- `configs/fp/predict.yaml` の `input.mode` に応じて
  - `formula`：CSVからCASを解決して推論（同一分子式で複数行ある場合は先頭を使用）
  - `cas`：CASを直接指定して推論

## 4) GNN学習/推論（任意）：`scripts/train.py`, `scripts/predict.py`

- 依存：`torch`, `torch_geometric`
- グラフ化：`src/gnn/featurizer_graph.py`
- モデル：`src/gnn/models.py`
- 成果物：`runs/train/gnn/<exp_name>/artifacts/model_best.pt` など

---

# 実行メモ

- macOS等では `python` が無い環境があるため、READMEの通り `python3 ...` を推奨します（venvを使う場合は `source .venv/bin/activate` 後に `python ...` でもOK）
- Matplotlibのキャッシュはプロジェクト内 `.cache/matplotlib` を自動使用するようにしてあります（書き込み不可なHOME環境でも動作しやすくします）


