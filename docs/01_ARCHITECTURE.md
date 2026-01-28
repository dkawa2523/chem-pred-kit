# Architecture（全体設計）

このドキュメントは、LJ_prediction を「物性予測の学習基盤」として拡張するための **設計の骨格** です。  
特に「Process（処理単位）」を中心に、拡張しやすく、比較評価しやすい設計を定義します。

---

## 1. コア設計：Process（処理単位）
### 1.1 なぜProcessを分けるか
- 前処理・特徴量化・学習・評価・可視化は改良頻度が高い
- “trainに追加の可視化を混ぜる”とスクリプトが肥大化し、比較や運用が壊れる
- 将来 ClearML で「処理ごとに Task」として管理するには、粒度が必要

### 1.2 Processの原則（不変）
- **1つのCLIエントリポイント = 1つのProcess**
- ProcessはHydraで管理し、単独実行できる
- Processは artifact を出し、他Processはそれを入力として参照できる

---

## 2. 推奨Process一覧（最小セット）
詳細は `docs/10_PROCESS_CATALOG.md` を正とする。

- **P01 build_dataset**  
  CSV/SDFを読み、欠損処理・フィルタなどを行い、学習可能なデータセット（内部表現 or キャッシュ）を構築する
- **P02 featurize**  
  特徴量を計算し、必要ならキャッシュする（高コスト計算を再利用）
- **P03 train**  
  学習（複数モデル/特徴量を config で切替可能）
- **P04 evaluate**  
  指標計算・比較可能な形式で出力（metrics/predictions）
- **P05 predict**  
  学習済みモデルで推論（trainと同じ pipeline をロード）
- **P06 visualize**  
  プロット生成（parity/residual/learning curve/AD可視化などを独立Processに）
- **P07 collect_data（任意）**  
  外部APIなどからデータ収集し、CSV/SDF形式へ整形（secrets厳守）

---

## 3. モデル/特徴量の拡張性（頻繁に変わる箇所）
### 3.1 plugin（registry）で増やす
- モデル/特徴量/タスクは registry に登録し、`name` で選ぶ
- if/elif 連鎖を増やさず、追加時は「新規クラス＋登録＋config追加」で完結

### 3.2 比較評価しやすい設計
- `configs/` で model/features/task を切り替え
- Hydra の `-m`（multirun）で一括比較を想定
  - 例: `python scripts/train.py -m model=mpnn,gcn features=morgan,rdkit`
- すべての run は `meta.json` と `metrics.json` を持ち、後から集計できる

---

## 4. ClearMLを見越した管理（今は実装しない）
- Process単位で artifact を出すことで、将来「Process = ClearML Task」に移行しやすくする
- `meta.json` に upstream 参照（どのデータセットrunを使ったか等）を残す
- ClearML依存コードは今は入れない（導入はADRで管理）

---

## 5. 依存方向（守る）
- scripts → src は OK
- src → scripts は NG
- RDKitや外部APIは `src/common` / `src/*/preprocess` / `src/*/features` に閉じ込める
