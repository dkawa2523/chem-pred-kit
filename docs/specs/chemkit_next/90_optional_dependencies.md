# Optional Dependencies（ChemKit Next）

## 問題
3D GNN/Transformer/Pretraining は追加ライブラリが必要になりやすく、
全ユーザーに強制すると環境が壊れやすい。

## 方針
- 依存は「extras」または「optional import」にする
- 実行時に必要な依存が無い場合は
  - 分かりやすいエラーメッセージ
  - 代替（2Dモデル/2D特徴）への fallback（可能なら）
- CI は base と optional を分ける（baseは軽く、optionalはnightly等）

## 例（メッセージ）
- `pip install -e .[3d]`
- `pip install -e .[transformer]`
- `pip install -e .[pretrain]`

## Install examples (this repo)
- `pip install -r requirements/base.txt`
- `pip install -r requirements/gnn.txt`
- `pip install -r requirements/optional-3d.txt`
- `pip install -r requirements/optional-transformer.txt`
- `pip install -r requirements/optional-pretrain.txt`

## このrepoのrequirements分離
- base: `requirements/base.txt`
- gnn(2D): `requirements/gnn.txt`（torchは環境に合わせて別途）
- optional 3d: `requirements/optional-3d.txt`
- optional transformer: `requirements/optional-transformer.txt`
- optional pretrain: `requirements/optional-pretrain.txt`

## TorchMD-Net（PaiNN backend）
`torchmd-net` はPyPIに無いため source install が必要です。`torch` と `torch_geometric` を先に入れてください。

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

モデル設定は `backend: torchmdnet` と `torchmd_model: equivariant-transformer` を指定します
（例: `configs/model/painn.yaml` / `configs/model/gnn_painn_quick.yaml`）。

### Troubleshooting
- `Invalid license` / `license field` エラー: `pyproject.toml` の `license = "MIT"` を
  `license = { text = "MIT" }` に修正して再インストール。
- コンパイル失敗（C++17不足 / macOSターゲット不一致）: `CFLAGS/CXXFLAGS` と
  `MACOSX_DEPLOYMENT_TARGET` を指定して再インストール。
- `ModuleNotFoundError: lightning_utilities`: `pip install -U lightning-utilities`。
- `ImportError: torchmdnet` が残る: 同じ仮想環境に入っているか確認し、
  `torch`/`torch_geometric` を先に揃えてから `pip install --force-reinstall` で再ビルド。
