# Task 011 (P0): smoke 用 fixture データセットの追加

## 目的（Why）
- raw CSV/SDF が無い環境でも最低限の smoke 実行ができるようにし、P0タスク（010/015/025）の受け入れ検証を可能にする。

## 背景（Context）
- 現状は `data/raw/` が空で、学習や監査の実行検証ができない。
- Hydra config 再整理・Process整理・監査Processの受け入れ条件が確認できず、タスクがblockedになっている。

## スコープ（Scope）
### In scope
- 最小の CSV + SDF fixture を追加（3〜10分子程度）
- fixture を参照する config を追加（dataset/fixture + train/evaluate/predict の最小構成）
- smoke で 1 パス通ることを確認

### Out of scope（やらない）
- 本番データの配布・ダウンロード自動化
- 高精度のモデル学習

## 影響（Contract Impact）
- Invariantsに抵触しない（Process追加なし）
- config/CLI の互換性は維持（fixture用の追加のみ）

## 実装計画（Plan）
1) `tests/fixtures/data/raw/` に最小 CSV と SDF を追加
2) `configs/dataset/fixture.yaml` と入口 config を追加
3) `configs/fp/train_fixture.yaml` など最低限の smoke 用 config を追加
4) `README` または `docs/` に fixture の使い方を追記（最小）

## 変更対象（Files）
- tests/fixtures/**
- configs/dataset/fixture.yaml
- configs/fp/train_fixture.yaml（必要なら gnn も）
- README.md / docs/README.md

## 受け入れ条件（Acceptance Criteria）
- [ ] fixture データで `scripts/train.py` が 1 パス通る
- [ ] fixture 用 config で `scripts/audit_dataset.py` が動く
- [ ] 010/015/025 の block が解除できる

## 検証手順（How to Verify）
- `python scripts/build_dataset.py --config configs/dataset_fixture.yaml`
- `python scripts/train.py --config configs/fp/train_fixture.yaml`
- `python scripts/audit_dataset.py --config configs/audit_dataset_fixture.yaml`

## メモ
- 分子は RDKit で生成可能な簡単なもの（例: ethanol, benzene, methane）を利用
