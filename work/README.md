# work/ の使い方（Codex開発の運用ルール）

`work/` は **都度の改良・開発指示（可変）** を置く場所です。  
1タスク=1ファイルにし、指示の混線を防ぎます。

## ルール
- 新規作業は `work/templates/` からコピーして `work/tasks/NNN_*.md` を作る
- Codexへは「SESSION_CONTEXT + Skill + Task」をまとめて渡す
- Done の定義はタスク内の Acceptance Criteria を満たすこと
- 仕様変更が必要なら `work/rfc/` に提案を書き、合意したら `docs/adr/` に移す
