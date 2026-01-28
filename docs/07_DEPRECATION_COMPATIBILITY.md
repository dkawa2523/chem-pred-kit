# Deprecation & Compatibility（互換性・非推奨）

このドキュメントは「設定・CLI・成果物の互換性」を守るための方針です。

## 1. 破壊的変更（Breaking Change）の定義
- config key の rename/remove
- CLI の主要引数の廃止
- artifact の必須ファイル/列の変更

## 2. 非推奨（Deprecation）プロセス
1) まず `DeprecationWarning` を出す（ログにも残す）
2) 1〜2リリース（または一定期間）維持
3) 影響範囲と移行手順を docs に追記
4) その後 remove（ADR必須）

## 3. バージョニング（提案）
- 研究用途なら日付ベースでもよいが、基盤化するなら SemVer 推奨
- major: breaking, minor: feature, patch: bugfix
