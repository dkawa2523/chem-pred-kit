# Invariants（不変条件）

このドキュメントは **開発中に変えてはいけない契約（Codexも遵守）** です。  
迷ったらここへ戻り、ここに書かれていない“勝手な最適化”をしないでください。

---

## 0. 基本目的（この基盤のゴール）
- **入力**: 化学構造（主に **CSV + SDF**。将来もこの形式を軸にする）
- **出力**: 物性（LJパラメータに限らず、**同じCSV/SDF形式のまま目的変数を差し替える**）
- **操作**: **CLI**（**Hydra**で設定管理）
- **要求**:
  - 前処理・特徴量化・学習・評価・推論・可視化を **拡張しやすい**
  - モデル/特徴量の手法が複数あり、**選択して実行・比較評価しやすい**

---

## 1. もっとも重要な不変条件（破るなら RFC→ADR→更新）

### 1.1 「処理単位（Process）」が第一級（最重要）
本基盤では **処理単位**を第一級の概念として扱う。

- 例: `build_dataset` / `preprocess` / `featurize` / `train` / `evaluate` / `predict` / `visualize` / `collect_data`
- 各処理は次を満たすこと：
  - **Hydra config により管理される**
  - **単独でCLI実行できる**
  - 入力・出力（artifact）が明確で、他処理から参照できる
  - “将来ClearML Task化できる粒度”である（後述）

> 原則：**1つのCLIエントリポイントは 1つのProcess**に対応させる。  
> 「trainのついでに可視化も全部やる」などの巨大化は避け、必要なら別Processに分ける。

### 1.2 設定が唯一の真実（Hydra中心）
- 実行挙動は **Hydra config** で決まる（ハードコードを増やさない）
- CLIの引数も、最終的には config の override として扱う
- パスや秘密情報は env で注入（docs/08参照）

### 1.3 学習-推論スキュー禁止（train/infer一致）
- **学習と推論で前処理・特徴量生成は同一実装/同一状態を使う**
- 学習時に fit したスケーラー/補完器/辞書などは artifact として保存し、推論でロードする
- 推論側で勝手に再fitしない

### 1.4 「比較可能性」を壊さない
複数手法の比較を可能にするため、以下を守る：

- dataset split（train/val/test）は再現可能に保存し、**同じsplitで比較**できること
- `metrics.json` / `predictions.csv` / `meta.json` の **キー・列の契約**を守る
- 乱数seedを管理し、比較がぶれないようにする（少なくともデフォルトseed）

### 1.5 成果物（artifact）の契約を破らない
- 各Processは `docs/04_ARTIFACTS_AND_VERSIONING.md` の契約に従って成果物を出す
- 形式変更が必要なら `work/rfc` → `docs/adr`

---

## 2. ClearML 連携を見越した不変条件（※今は実装しない）
将来的に、各Processを ClearML の **Task** として扱う想定のため、以下を守る：

- 各Processは「入力（上流artifact参照）」「出力（artifact）」を明示する
- すべてのパラメータは config から決定できる（対話入力禁止）
- `meta.json` に次を含める（今から入れておく）：
  - `process_name`, `run_id`, `dataset_hash`, `config_hash`, `git_sha`（可能なら）
  - `upstream_artifacts`（参照した上流runのパス/ID）
  - `tags`（任意、将来ClearMLでフィルタに使う）

---

## 3. 変更して良い範囲（ただし上記契約を守る）
- 新しい物性（目的変数）追加（同じCSV/SDF形式）
- 特徴量追加（FP/記述子/3D/埋め込み）
- 新モデル追加（FP/GNN/事前学習など）
- 評価指標・可視化追加（評価の比較可能性を維持すること）
- データ収集機能（外部API連携。ただし secrets 管理は厳守）

---

## 4. 変更の進め方（必須）
- 変更は `work/tasks/NNN_*.md` のタスク単位で行う
- Processの追加・分割・統合は `docs/10_PROCESS_CATALOG.md` に反映する
- 仕様変更は `work/rfc` → `docs/adr` で記録する
