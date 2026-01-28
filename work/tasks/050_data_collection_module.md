# Task 050 (P1): データ収集モジュールの拡張設計（外部API）

## 目的
- 物性の追加に伴い、用途別のデータ収集機能を拡張可能にする

## 方針
- secrets は env 参照（docs/08）
- 収集ロジックは plugin 化（data_source registry）

## Plan
1) `src/data_collection/`（案）を作る（APIクライアント層/整形層/キャッシュ層）
2) まずは “スタブ” 実装（ダミーデータ）で全体I/Fを決める
3) 後から PubChem 等の実装を追加できる形にする

## Acceptance Criteria
- [x] data_source を切替可能（config）
- [x] secrets をコードに埋め込まない

## Result
- data_source は `configs/data_source/*` と `configs/data_collection/default.yaml` で切替可能
- secrets は `api_key_env` 参照のみ（コード埋め込み禁止）
