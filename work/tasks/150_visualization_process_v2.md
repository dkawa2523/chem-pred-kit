# Task 150 (P0): Visualization Process v2（parity/residual）

## 目的
新データセット・新モデルでも共通の可視化を出し、比較を高速化する。

## In scope
- visualize process の拡張（parity/residual/hist）
- multi-target の場合：代表ターゲット or 全ターゲットのどちらを出すか設定化
- artifactに plots/ を保存

## Acceptance Criteria
- [x] predictions.csv から parity/residual/hist を生成できる（synthetic fixture run）
- [x] contract に違反しない（plots/ 配下に追加ファイル）

## Verification
```sh
python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

out_dir = Path("runs/evaluate/fixture_fp_evaluate")
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
rows = []
for i in range(60):
    split = "train" if i < 30 else "val" if i < 45 else "test"
    y_true = float(rng.normal(loc=0.0, scale=1.0))
    y_pred = y_true + float(rng.normal(loc=0.0, scale=0.2))
    rows.append({"split": split, "y_true": y_true, "y_pred": y_pred})

pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
metrics = {
    "by_split": {
        "train": {"mae": 0.12, "rmse": 0.2, "by_target": {"target": {"mae": 0.12, "rmse": 0.2}}},
        "val": {"mae": 0.25, "rmse": 0.35, "by_target": {"target": {"mae": 0.25, "rmse": 0.35}}},
        "test": {"mae": 0.3, "rmse": 0.4, "by_target": {"target": {"mae": 0.3, "rmse": 0.4}}},
    }
}
with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
PY
python scripts/visualize.py --config configs/fp/visualize_fixture.yaml
pytest tests/test_visualize_utils.py
```
