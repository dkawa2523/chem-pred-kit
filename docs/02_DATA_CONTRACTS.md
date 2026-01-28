# Data Contracts（データ契約）

このドキュメントは **CSV/SDF を中心としたデータ形式の共通契約** を定義します。  
目的変数（物性）が増えても、ここを守れば基盤が壊れないようにします。

## 1. 入力データの基本形（推奨）
### 1.1 CSV（テーブル）
- 1行 = 1分子（サンプル）
- 必須列（推奨名）
  - `sample_id`：一意ID（無ければ生成）
  - `cas`：CAS番号（ある場合）
  - `smiles`：SMILES（ある場合）
  - `formula`：分子式（ある場合）
  - `sdf_path`：SDF ファイルへのパス（行ごと or 共通）
  - `target.<name>`：目的変数（例 `target.lj_sigma`, `target.lj_epsilon`）
- 任意列
  - `meta.*`：データソース、温度条件、参考値など

※ 既存CSVの列名が異なる場合は `configs/dataset/*.yaml` でマッピングする。

### 1.2 SDF（構造）
- RDKit で読めること
- 可能なら 3D 座標を含む（無い場合は preprocess で生成する戦略も可）
- 分子識別子は `CAS` や `PUBCHEM_COMPOUND_CID` 等が SDF property に入っている場合がある
  → dataset loader が CSV との突合を担当する

## 2. 内部共通表現（BaseSample）
`src/data` が返す内部表現は、最低限次を満たすこと：
- `id: str`
- `mol: RDKit Mol | None`（SDF/SMILES から生成）
- `targets: dict[str, float | int | None]`
- `meta: dict[str, Any]`

GNN の場合はこの後 `graph`（PyG Data 等）へ変換されてもよいが、
**Mol を捨てない**（後段の特徴量や可視化で使える）。

## 3. 分割（split）契約
- `split.train/val/test` の index を保存して再現可能にする
- split は “方法” を明記する（random/scaffold/time 等）
- split 作成時は `dataset_hash` を記録し、データ更新時に検知する

## 4. 欠損・異常値ルール
- 目的変数が欠損の場合：
  - 学習対象外にする（デフォルト）
  - ただしマルチタスクでは「タスクごとに欠損許容」を設定できる
- Mol が生成できない場合：
  - 原則として除外し、除外理由をログ/レポートに残す
  - 将来：修復（sanitize/tautomer）を preprocess として実装可能

## 5. 目的変数追加の手順（同じCSV/SDFでも物性が増える場合）
1) `configs/task/<property>.yaml` を追加し、`target.*` の列を指定
2) `src/tasks/` にタスク定義（ロス・メトリクス）を追加（必要なら）
3) `work/tasks/*_add_property_*.md` のテンプレに従い、最小 smoke test を追加
