# AUTO Mode（タスク/スキル指定なしで進める）

このファイルは「ユーザーがタスクやスキルを毎回指定するのが面倒」という問題を解決するための **自動ルーティング** です。

## あなた（Codex）が最初に読むべきファイル
1) docs/00_INVARIANTS.md
2) work/queue.json
3) agentskills/skill_registry.json
4) agentskills/ROUTER.md
5) codex/PROMPT_RULES.md
6) codex/CHECKLIST.md

## 自動実行ルール
### 1. 次にやるタスクの決定
- work/queue.json の `tasks` から、`status == "todo"` のものを **priority順（P0→P1→P2）** に選ぶ
- 同priority内は `id` の昇順で選ぶ
- 選んだタスクを `in_progress` に変更し、work/queue.json を更新する（auto_set が true の場合）

### 2. 参照すべきスキルの決定
- タスクに `skills` があれば、そのスキルを採用する
- 無い場合は agentskills/ROUTER.md を参照し、タスクの内容から適切なスキルを選ぶ
- 採用したスキルカード（agentskills/skills/*.md）を読んで手順に従う

### 3. 参照すべき Contract（docs）を決定
- タスクに `contracts` があれば必ず読む
- 無ければ docs/00_INVARIANTS.md を必ず読む

### 4. 出力フォーマット（必須）
あなたの返答は次の順に出す：
1) 選択したタスク（id/title/path/priority）
2) 採用したスキル一覧
3) 読むべき Contract 一覧
4) 変更計画（ファイル単位）
5) 実装（差分が分かるように）
6) 追加/更新したテスト
7) 検証コマンド
8) 互換性への影響（config/CLI/artifact）

### 5. 完了時のステータス更新
- タスクの Acceptance Criteria を満たしたら status を `done` に更新し、work/queue.json を保存
- もしブロックされたら `blocked` にし、理由と次の一手を `work/tasks/<task>.md` に追記

---

✅ ユーザーはこの AUTO.md の内容を一度貼る（または「AUTOモードで」と伝える）だけで、
以後は優先度順にタスクを選び、適切なスキルと契約を参照して進められます。


---

## 追加ルール（Process中心の設計を守る）
- 変更が “別Processに切り出すべき” 内容（可視化追加、比較集計追加等）の場合、
  既存scriptに無理やり詰めず **新しいProcessとして scripts/* を追加**することを優先する
- Process追加/分割を行ったら `docs/10_PROCESS_CATALOG.md` を更新する
- 将来ClearML Task化を想定し、各Processは単独実行できることを守る


---

## BLOCKED の扱い（重要）
- queueに blocked がある場合、次へ進む前に **解除（分割/条件明記）**を優先する
- blocked のままにする場合は、必ず task md に reason/unblock_condition/next_action を追記し、解除用の小タスクを queue に追加する


## 解除タスクの命名・リンク規約（重要）
- blocked を解除するためのサブタスクを作る場合：
  - 新規タスクmdに `Unblocks: <親ID>` を書く（例: `Unblocks: 010`）
  - 親タスクの Blocked セクションに `unblock_tasks: [<子ID>]` を追記する（例: `unblock_tasks: [011]`）
  - work/queue.json に子タスクを追加する
- tools/codex_prompt.py はこの情報を使って、blocked 親タスクではなく解除サブタスクを優先提示します。


## 追加ルール（v3.3）
- blocked解除用の子タスクを作る場合、work/queue.json の子タスクに **unblocks: ["<parent_id>"]** を必ず付ける。
- 依存関係がある場合、depends_on を付け、todoの順序事故を防ぐ。
