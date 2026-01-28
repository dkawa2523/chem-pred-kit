# Task 107 (P2): PCQM4Mv2 Adapter（Pretrain用途）

## 目的
大規模データ（PCQM4Mv2）を「pretrain用dataset」として取り込む。

## 注意
- サイズが大きく、CIでは扱わない（ローカル/専用環境で）
- 取得は OGB 経由が現実的

## In scope
- adapter: pcqm4mv2
- `target.gap` を canonical 名に揃える（HOMO-LUMO gap）
- pretrain config（エポック/バッチ制限）を用意

## Acceptance Criteria
- [x] download/prepare ができる
- [x] pretrain モードで1エポック回る

## 検証
- `pytest tests/contract/test_pcqm4mv2_adapter_import.py`
- `pytest tests/contract/test_pcqm4mv2_adapter_contract.py`
