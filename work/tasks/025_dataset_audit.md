# Task 025 (P0): データ監査＆漏洩/重複検知（比較の信頼性確保）

## 目的（Why）
- R²などの比較が「本当に意味のある比較」になるように、データの地雷（重複/漏洩/単位混在/ラベル異常/無効構造）を可視化して潰す。
- “精度改善の前に、評価が正しいことを保証する” がP0。

## 背景（Context）
- 1万件規模でR²が低い原因は、モデル以前に「データ問題」であることが多い。
- split漏洩（同一分子がtrainとtestに混入等）があると、比較が壊れる。
- 将来ClearMLで Process を Task 化するため、監査も独立Process化する。

## スコープ（Scope）
### In scope
- **新Process `audit_dataset`** を追加（1 script = 1 process）
  - 例: `scripts/audit_dataset.py`
- auditの結果を **artifactとして保存**
  - `audit_report.json`（機械可読）
  - `audit_report.md`（人間可読）
  - `plots/`（分布/外れ値など）
- 監査項目（最低限）
  - 構造の妥当性（RDKitでMol生成できない行）
  - **重複検知**（canonical SMILES / InChIKey）
  - ターゲット欠損/異常（NaN、極端値、分布）
  - splitがある場合：**split間重複/漏洩検知**
  - 主要統計（件数、元素種分布、分子量分布、ターゲット統計）

### Out of scope（今回はやらない）
- 自動修復（塩除去や中性化で直す等）を全面的にやる
  - ただし「修復候補の検出」はOK
- ClearML SDK の導入（設計はdocs/12に従うが実装はしない）

## 影響（Contract Impact）
- Invariantsに抵触しない（Process追加は推奨）
- artifact契約に追加（auditの成果物）→ **docs/10_PROCESS_CATALOG.md を更新する**

## 実装計画（Plan）
1) `docs/10_PROCESS_CATALOG.md` に `audit_dataset` を追記（入力/出力を明記）
2) `scripts/audit_dataset.py` を追加
   - 入力：
     - raw CSV/SDF 直読み OR `build_dataset` artifact（どちらか、configで選択）
   - 出力：
     - `audit/audit_report.json`
     - `audit/audit_report.md`
     - `plots/*.png`
3) `src/data/audit.py`（案）に監査ロジックを分離
   - canonical SMILES、InChIKeyの計算
   - 重複クラスタの抽出
   - split漏洩チェック（同一キーが複数splitに存在）
4) `configs/process/audit_dataset.yaml`（入口）と `configs/audit/default.yaml`（詳細）を追加
5) テスト追加
   - 小さなCSV（数件）で audit が走り、reportに必須キーが入ること

## 受け入れ条件（Acceptance Criteria）
- [ ] `python scripts/audit_dataset.py ...` が単独で実行できる（Hydra管理）
- [ ] `audit_report.json` に最低限のキーが入る（例：invalid_mol_count, duplicate_groups, target_stats）
- [ ] splitがある場合、split漏洩が検知できる（0件でもOK）
- [ ] `docs/10_PROCESS_CATALOG.md` が更新されている
- [ ] pytestで最低1つテストが追加されている

## 検証手順（How to Verify）
- 例：
  - `python scripts/audit_dataset.py dataset=quick audit=default`
  - `pytest -q`

## メモ
- 監査結果を見て次タスクを起票する（例：重複排除、単位統一、外れ値処理など）
