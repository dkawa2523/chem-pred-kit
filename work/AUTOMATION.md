# タスク/スキル指定の手間を減らす（AUTO運用）

## 目的
- 毎回「このタスク」「このスキル」をプロンプトに書くのが面倒 → 自動化します。

---

## 方法A：Codex AUTO（最小手間）
1) `codex/AUTO.md` を一度貼る（または「AUTOモードで」と指示）
2) Codex が `work/queue.json` を読み、優先度順にタスクを選ぶ
3) Codex が必要なスキルカード・契約 docs を開いて進める
4) 終わったら `work/queue.json` を done に更新

---

## 方法B：ローカルで “貼り付け用プロンプト” を生成（安定）
```bash
python tools/codex_prompt.py list
python tools/codex_prompt.py next
```

- `next` 実行で自動的に in_progress に変わります。
- 手動で done にしたい場合：
```bash
python tools/codex_prompt.py done 010
```


## 追加メモ
- P0では「Process単位で独立実行」「artifact契約」「train/infer skew排除」を最優先にします。


## 重要（v3.1）: blocked をスキップしない
- `python tools/codex_prompt.py next` は **in_progress → blocked → todo** の順に次を選びます。
- blocked を例外的に飛ばしたい場合のみ `python tools/codex_prompt.py next --skip-blocked` を使います（推奨しません）。


## blockedをスルーしない運用（v3.2）
- next は `in_progress -> blocked -> todo` で選びます。
- ただし blocked 親タスクに `unblock_tasks` が書かれている場合、解除サブタスク（todo/in_progress）を優先して提示します。
- 解除サブタスクには `Unblocks: <親ID>` を書くと、検出が安定します。


## 追加: doctor（キュー整合性チェック）
```bash
python tools/codex_prompt.py doctor
```
- 欠落タスクmd、重複ID、孤立blocked、欠落depends_onを検出します。

## 追加: unblocks / depends_on フィールド（推奨）
- 子タスクが親blockedを解除する場合、queue.json の子タスクに `unblocks: ["010"]` を付けると next が迷いません。
- 依存関係がある場合、`depends_on: ["020"]` のように書くと、順番事故が減ります。
