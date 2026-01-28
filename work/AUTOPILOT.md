# Autopilot: Codex + codex_prompt の自動ループ実行

このリポジトリの開発は `work/queue.json` のタスクを `tools/codex_prompt.py next` で選び、
そのプロンプトを Codex に貼って実行する想定です。

ここでは、**その反復を自動化**する方法をまとめます。

> ⚠️ 注意:
> - 完全自動化は便利ですが、誤編集・誤コマンド実行のリスクが上がります。
> - まずは `--full-auto` ではなく **sandbox/approval を厳しめ**にして試し、慣れてから自動化範囲を増やしてください。

---

## 推奨: Codex CLI の `codex exec` を使う

Codex CLI は `codex exec` で非対話実行でき、**PROMPT に `-` を指定すると stdin からプロンプトを受け取れます**。

最小ワンライナー:

```bash
python tools/codex_prompt.py next | codex exec --full-auto -
```

---

## ループ実行（Bash）

`tools/autopilot.sh` を使うと、doctorチェック→next→codex exec を繰り返せます。

```bash
bash tools/autopilot.sh 20
```

- 引数: 最大反復回数（省略時 20）
- `work/.autopilot/` 配下にログを保存します

### 主要な環境変数

- `CODEX_FLAGS` : codex exec に渡す追加フラグ
  - 例: `CODEX_FLAGS="--sandbox workspace-write --ask-for-approval on-request"`
- `CODEX_JSON=1` : `--json` でイベントJSONLを保存

---

## ループ実行（Python）

`tools/codex_autopilot.py` は bash より頑健（同一タスク連続など）に止められます。

```bash
python tools/codex_autopilot.py --max-iters 50 --full-auto
```

---

## 典型的な止まり方

- `doctor` が失敗 → キューやタスクmdが壊れている（修正してから再開）
- `codex exec` が非0 → Codexが失敗（stderrログを確認）
- `No tasks available` → キューが完了（または todo が無い）

---

## 安全な運用（推奨）

- **git feature branch** で実行し、開始前は `git status` をクリーンにする
- 反復ごとに `git diff` とテストを走らせる（必要なら script に追加）
- `--dangerously-bypass-approvals-and-sandbox`（`--yolo`）は原則使わない


## v3.7 重要: Codex CLI フラグの並び
- `--ask-for-approval` は CLI 実装によっては **グローバルフラグ**として扱われ、`codex exec` の後ろに置くとエラーになります。
- 一方 `--sandbox` は `codex exec` のフラグとして実装されていることが多く、`exec` の後ろに置く必要があります。
- 本キットの `tools/autopilot.sh` は `CODEX_GLOBAL_FLAGS` / `CODEX_EXEC_FLAGS` に分けて正しい順で実行します。

## codex_prompt の autopilot mode
`python tools/codex_prompt.py next --mode autopilot` を使うと、
- ユーザー確認を求めない
- queue.json の status を single source of truth として扱う
- blocked は解除タスク起票か status 更新で必ず前に進める
という “止まりにくい” ルールでプロンプトを生成します。



## v3.6 追加: “止まりやすい原因” への自動対処

`tools/autopilot.sh` は、連続実行で起きがちな停止要因を検出して対処します。

### 1) doctor 失敗（blockedに解除タスクが無い / md欠落 / depends_on欠落）
- `AUTO_FIX_DOCTOR=1`（デフォルト）なら、doctor 出力を Codex に渡して **整合性修復だけ**を自動で試みます。

### 2) 「確認お願いします」「未実装」「差分なし」で止まる
- `FORCE_PROGRESS_ON_STALL=1`（デフォルト）なら、これらの停止文言を検出したときに
  **“確認禁止・前進必須” の追いプロンプト**を追加で流し、次のいずれかを必ずさせます：
  - 実装して done
  - blocked + 解除子タスク（unblocks付き）を起票
  - queue/task.md の整合を取ってから実装

### 3) 無限ループ防止（同一タスク/無変更の連続）
- `MAX_SAME_TASK`（デフォルト3）: 同一タスクが連続しすぎると停止します
- `MAX_NO_CHANGE`（デフォルト2）: queue hash と git diff が連続で変わらない場合に停止します

停止した場合は `work/.autopilot/<timestamp>/` のログを開き、
- `prompt_*.md`
- `codex_out_*.txt`
- `codex_err_*.txt`
を確認して、blocked/依存関係/承認周りを調整してください。

---

## 便利な環境変数（例）

### まずは安全寄り（止まるならログで原因特定）
```bash
STOP_WHEN_ONLY_BLOCKED=1 REQUIRE_CLEAN_GIT=1 ./tools/autopilot.sh 10
```

### ガンガン回す（ただしレビュー前提）
```bash
FORCE_PROGRESS_ON_STALL=1 AUTO_GIT_COMMIT=1 ./tools/autopilot.sh 30
```
