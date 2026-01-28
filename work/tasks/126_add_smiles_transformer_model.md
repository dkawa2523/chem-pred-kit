# Task 126 (P1): SMILES Transformer Baseline（ChemBERTa等）

## 目的
SMILES Transformer（ChemBERTa/MolFormer）を導入し、文字列ベースの強いベースラインを作る。

## 依存
- transformers / tokenizers（optional dependency 推奨）

## In scope
- model=chemberta（仮） wrapper（encode SMILES → pooled → head）
- tokenizer/encoding を FeatureSet として扱い、skew排除（fit stateは基本なし）
- fine-tune と linear-probe の両方に対応する足場（Task 140で共通化）

## Acceptance Criteria
- [ ] QM9 small で train→eval が通る
- [x] 依存が無い場合は明確なエラー

## Done
- SMILES tokenizer/model の optional dependency エラーメッセージをテストで検証

## Verification
- `pytest tests/test_smiles_optional_deps.py`
