# Hydra Config Conventions（設定規約）

このドキュメントは Hydra を用いた設定管理の規約です。  
**Process（処理単位）をHydraで統一管理**し、手法の切替・比較評価をしやすくします。

---

## 1. 設定のゴール
- **各Processが単独で実行可能**
- モデル/特徴量/タスクの組合せを安全に切替可能
- 比較評価（multirun）しても成果物が追跡可能
- 将来ClearMLで Task 化しても、同じconfigを使い回せる

---

## 2. config group 構造（必須）
- `process/`：どのProcessを実行するか（脚本ごとに固定でもよいが、共通鍵は持つ）
- `dataset/`：入力データ・列マッピング・split・フィルタ
- `task/`：目的変数/タスクタイプ（regression/classification/multitask）
- `preprocess/`：正規化/3D生成/欠損処理
- `features/`：FP/記述子/3D/埋め込み（**頻繁に増える**）
- `model/`：FPモデル/GNNモデル/事前学習（**頻繁に増える**）
- `train/`：optimizer/scheduler/epochs/batch/seed
- `eval/`：メトリクス/プロット
- `infer/`：推論バッチ/出力形式
- `hydra/`：出力dir/sweeper/launcher

---

## 3. Processごとの “入口config” を用意する（推奨）
例：
- `configs/process/train.yaml`
- `configs/process/evaluate.yaml`
- `configs/process/predict.yaml`

ただし実装は自由（`scripts/train.py` に固定でもよい）。  
重要なのは **同じキー構造でI/Oとartifactが決まる**こと。

---

## 4. run名と出力ディレクトリ（比較可能性のため）
- `experiment.name` を持つ（ユーザーが意味のある名前を付けられる）
- 出力dirには少なくとも `process`, `experiment.name`, `timestamp` を含める
- `hydra.job.name` は process 名に寄せる（ログやClearML移行で便利）

---

## 5. multirun（比較評価）の規約（推奨）
- 比較はできるだけ **同一dataset split** で行う
- `train.seed` を固定して比較する（必要なら seed sweep も可能）
- 例（概念）:
  - `python scripts/train.py -m model=mpnn,gcn features=morgan,rdkit train.seed=0`
- 出力は run ごとに `meta.json`/`metrics.json` が残るので、後から集計する

---

## 6. バリデーション（必須）
起動時に config を検証する（例）：
- 3D必須モデルなのに 3D が無い → エラーで停止
- classificationなのに回帰lossが選ばれている → エラー

バリデーションは “静かなバグ” を防ぐ最重要ポイント。
