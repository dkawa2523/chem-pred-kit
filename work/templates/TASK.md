# TASK テンプレート

# Task: <タイトル>

## 目的（Why）
- 何を解決するか（ユーザー価値 / 研究価値 / 運用価値）

## 背景（Context）
- 現状の課題
- 既存実装/設定の所在（ファイルパス）

## スコープ（Scope）
### In scope
- ...

### Out of scope（やらない）
- ...

## 影響（Contract Impact）
- Invariantsに抵触する？（Yes/No）
- config/CLI/artifact 互換に影響？（Yes/No）
- 影響があるなら移行方針

## 実装計画（Plan）
1) ...
2) ...
3) ...

## Links（Optional: 依存関係メモ）
- Unblocks: <タスクID>   # このタスクが解除する blocked 親タスク（例: 010）
- Blocked By: <タスクID> # このタスクが依存している blocked 親タスク
- Related: <タスクID,...>

## Blocked（Optional: ブロック管理）
- reason: （なぜ止まっているか）
- unblock_condition: （何が揃えば再開できるか）
- next_action: （次の一手。多くの場合「解除用の小タスクを作る」）
- unblock_tasks: [<タスクID>, ...]  # 解除用サブタスク（例: [011]）

## 変更対象（Files）
- src/...
- configs/...
- scripts/...
- tests/...

## 受け入れ条件（Acceptance Criteria）
- [ ] 期待する CLI が動く
- [ ] smoke test が追加/更新された
- [ ] artifact が契約どおり生成される
- [ ] ドキュメント更新が必要なら実施した

## 検証手順（How to Verify）
コマンド例：
- `pytest -q`
- `python scripts/train.py ...`
- `python scripts/predict.py ...`

## メモ
- 参考文献/参考実装
