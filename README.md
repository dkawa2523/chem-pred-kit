# LJ MI Pipeline (Tc/Pc/Tb -> Lennard-Jones, SDF -> ML)

このリポジトリは、以下を「設定駆動（YAML）」で一貫して実行するためのベース実装です。

- **データセット作成**：`Tc, Pc, Tb`から **Lennard–Jones (LJ) パラメータ** `epsilon/k` と `sigma` を推算し、CSVへ列追加
- **疎な元素分布に対するデータ選別**：元素ホワイトリスト・希少元素除外・サイズ制限・分布平坦化・多様性サンプリング等を **プラグイン的に切替**
- **学習2（Fingerprint系）**：Morgan/ MACCS/ RDKit FP + (任意) RDKit記述子 → RF / CatBoost / LightGBM / GPR
- **学習1（GNN系）**：SDF → 分子グラフ（PyTorch Geometric） → GCN / MPNN
- **推論**：分子式（Hill式）またはCAS入力 → LJ推論 + **外挿（Applicability Domain）指標**とWarning

> **運用も視野**：`runs/` 配下に実験成果物（モデル、前処理器、設定スナップショット、評価図）を保存し、他エージェントでも追跡可能な構成にしています。

---

## 1) ディレクトリ構成

```
data/
  raw/
    tc_pc_tb_pubchem.csv
    sdf_files/            # CAS.sdf が入る
  processed/
    dataset_with_lj.csv
    indices/              # splitのindex
    cache/                # 特徴量キャッシュ
configs/
  dataset.yaml
  fp/train.yaml
  fp/predict.yaml
  gnn/train.yaml
  gnn/predict.yaml
runs/
  train/fp/<exp_name>/
  train/gnn/<exp_name>/
  evaluate/<exp_name>/
  predict/<exp_name>/
  visualize/<exp_name>/
src/
  common/                 # LJ計算、選別、split、ADなど
  fp/                     # Fingerprint学習/推論
  gnn/                    # GNN学習/推論
scripts/
  build_dataset.py
  train.py / evaluate.py / predict.py / visualize.py
```

---

## 2) セットアップ

### 必須（Fingerprint学習まで）
- Python 3.10+ 推奨
- `rdkit`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `lightgbm`, `catboost`, `pyyaml`, `tqdm`

推奨（venv + requirements）：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

例（環境に合わせて調整してください）：
```bash
pip install -U numpy pandas scikit-learn matplotlib pyyaml tqdm lightgbm catboost
# RDKitはcondaが一般的
# conda install -c conda-forge rdkit
```

### 追加（GNN学習）
- `torch`, `torch_geometric`（PyTorchとCUDA/CPUに整合するwheelを入れてください）

PyGの導入は公式手順が確実です：
- https://pytorch-geometric.readthedocs.io/

### 追加（PaiNN / TorchMD-Net バックエンド）
`torchmd-net` はPyPIに無いため source install します。`torch`/`torch_geometric` は先に入れてください。

macOS (CPU build) の例:
```bash
pip install -U wheel lightning-utilities
git clone https://github.com/torchmd/torchmd-net.git work/torchmd-net

# pyproject.toml の license 形式エラーを回避するパッチ
python3 - <<'PY'
from pathlib import Path
path = Path("work/torchmd-net/pyproject.toml")
text = path.read_text(encoding="utf-8")
text = text.replace('license = "MIT"', 'license = { text = "MIT" }')
path.write_text(text, encoding="utf-8")
PY

ACCELERATOR=cpu \
MACOSX_DEPLOYMENT_TARGET=11.0 \
CFLAGS="-O3 -std=c++17 -mmacosx-version-min=11.0" \
CXXFLAGS="-O3 -std=c++17 -mmacosx-version-min=11.0" \
pip install --no-deps --no-build-isolation --force-reinstall work/torchmd-net
```

設定側は `configs/model/painn.yaml` / `configs/model/gnn_painn_quick.yaml` で
`backend: torchmdnet` と `torchmd_model: equivariant-transformer` を使います。

---

## 3) データ配置

1. `tc_pc_tb_pubchem.csv` を `data/raw/tc_pc_tb_pubchem.csv` に配置  
2. SDFを `data/raw/sdf_files/` に配置  
   - ファイル名は **CAS列と一致**させてください（例：`71-43-2.sdf`）

---

## 4) 実行手順（Quickstart）

開発/スモークテスト用の軽量設定：

- `configs/dataset_quick.yaml`
- `configs/fp/train_quick.yaml`
- `configs/fp/predict_quick.yaml`
- `configs/gnn/train_quick.yaml`

### Step 1: データセット作成（LJ列追加 + 選別 + split）
```bash
python3 scripts/build_dataset.py --config configs/dataset.yaml
# デバッグ用（先頭N件だけ）
# python3 scripts/build_dataset.py --config configs/dataset.yaml --limit 500
```

- 出力：
  - `data/processed/dataset_with_lj.csv`
  - `data/processed/indices/{train,val,test}.txt`
  - `data/processed/dataset_config_snapshot.yaml`

### Step 2: 学習2（Fingerprint）
```bash
python3 scripts/train.py --config configs/fp/train.yaml
```

- 出力：
  - `runs/train/fp/<exp_name>/artifacts/model.pkl`
  - `runs/train/fp/<exp_name>/artifacts/imputer.pkl`（欠損対策）
  - `runs/train/fp/<exp_name>/artifacts/ad.pkl`（外挿診断用）
  - `runs/train/fp/<exp_name>/plots/*.png`（Parity/Residual/学習曲線等）

### Step 3: 推論（Fingerprint）
```bash
python3 scripts/predict.py --config configs/fp/predict.yaml --query C6H6
# または
python3 scripts/predict.py --config configs/fp/predict.yaml --query 71-43-2
```

- 推論時に以下を表示します：
  - 予測値
  - 近傍類似度（Tanimoto）
  - Trust Score（0〜100）
  - Warning（外挿の可能性）

### Step 4: 学習1（GNN: PyTorch Geometric）
```bash
python3 scripts/train.py --config configs/gnn/train.yaml
```

### Step 5: 推論（GNN）
```bash
python3 scripts/predict.py --config configs/gnn/predict.yaml --query C6H6
```

---

## 5) データ選別（疎な元素分布への対処）

`configs/dataset.yaml` の `selectors:` を上から順に適用します。

実装済みセレクタ/サンプラ：
- `element_whitelist`：許可元素のみ残す
- `element_blacklist`：禁止元素を含むものを除外
- `min_element_frequency`：出現頻度が少ない元素を含む分子を除外
- `max_size`：`n_heavy_atoms` 等でサイズ上限
- `target_range`：目的変数の妥当範囲外を除外
- `target_stratified`：目的変数分布を均してダウンサンプル
- `diversity_farthest_point`：Tanimoto空間で多様性最大化サンプリング
- `butina_cluster`：Butinaクラスタリングでクラスタごとにサンプリング

> 推奨：まず **有機系に限定（whitelist）→希少元素除外→サイズ制限**で安定化し、その後に多様性サンプリングや分布平坦化を試すと回しやすいです。

---

## 6) LJ推算式について

`src/common/lj.py` に実装しています。デフォルトは Bird, Stewart, Lightfoot のCSP相関に準拠：

- `epsilon/k = 0.77 * Tc`
- `sigma = 2.44 * (Tc / Pc_atm)^(1/3)`  ※ `Pc`は入力がPaで、内部でatmへ換算

他の相関は `method` を追加して拡張できます。

---

## 7) 拡張ポイント（VSCode/Codexでの改良向け）

- **LJ計算式の追加**：`src/common/lj.py`
- **選別手法の追加**：`src/common/dataset_selectors.py`
- **split法の追加**：`src/common/splitters.py`
- **前処理の追加**（中性化、塩除去、3D生成など）：`src/common/chemistry.py` + 各featurizer
- **GNNモデル追加**（GIN/GAT/3Dモデル等）：`src/gnn/models.py`
- **不確かさ推定**（MC Dropout / ensemble / conformal）：`src/common/ad.py` + train/predict

---

## 8) トラブルシューティング

- `sdf_dir not found`：`data/raw/sdf_files/` のパスとファイル名（CAS一致）を確認
- `torch_geometric` が無い：GNN系はPyGが必要。公式手順でインストール
- GNN学習が「止まったように見える」：`torch_scatter` / `torch_sparse` が無いと **極端に遅く**なることがあります（`train.log` に警告が出ます）
- `MPS backend out of memory`：`configs/gnn/train.yaml` の `train.batch_size` / `model.hidden_dim` / `model.num_layers` / `model.edge_mlp_hidden_dim` を下げる、または `train.device: cpu` にする
- 分子式からCAS解決が曖昧：同一分子式が複数行あり得ます。**運用ではCAS入力推奨**

---

## 9) 開発メモ

- **設定スナップショット**は各`run_dir`に保存しています（再現性）
- Fingerprint学習は `data/processed/cache/fp/` に特徴量キャッシュを作ります（同設定なら再利用）

---

## 10) 開発フロー（devkit）

- 不変条件/契約は `docs/00_INVARIANTS.md` と `docs/README.md` を起点に確認
- タスク運用は `work/tasks/` と `work/queue.json`（一覧は `work/BACKLOG.md`）
- Codex向け運用は `codex/README.md` と `codex/SESSION_CONTEXT.md`
- Skills と現状マップは `agentskills/README.md` と `work/REPO_ORIENTATION.md`

---

## 11) Fixture dataset（smoke用）

最小データでの smoke 実行は以下を使用します：
- CSV: `tests/fixtures/data/raw/tc_pc_tb_fixture.csv`
- SDF: `tests/fixtures/data/raw/sdf_files/`

例：
```bash
python scripts/build_dataset.py --config configs/dataset_fixture.yaml
python scripts/train.py --config configs/fp/train_fixture.yaml
python scripts/audit_dataset.py --config configs/audit_dataset_fixture.yaml
```
