# ChemKit Next MD Pack: 既存 test_chemkit へのマージ手順

このZIPは **既存のディレクトリに“追加”**できる形で作っています（基本は上書き不要）。

## 1) 追加されるもの（新規）
- `docs/specs/chemkit_next/*`
- `docs/policies/chemkit_next/*`
- `work/tasks/100_*` など（100〜169の範囲）
- `work/codex/chemkit_next/*`
- `work/agentskills/skills/chemkit_next/*`
- `work/queue.append.chemkit_next.json`
- `tools/queue_merge_append.py`
- `tools/skills_merge_append.py`

## 2) マージ手順（推奨）
リポジトリルートで：

```bash
# 1. zipを repo root に展開（docs/ や work/ が直下に来る想定）
unzip -o chemkit_next_mdpack_v1.zip -d .

# 2. queue.json に新タスクを“追記”する（上書きしない）
python tools/queue_merge_append.py --queue work/queue.json --append work/queue.append.chemkit_next.json

# 3. 新しいskillsを registry に追記（任意。既存skillだけで回すなら不要）
python tools/skills_merge_append.py --registry work/agentskills/skill_registry.json --append work/agentskills/skill_registry.append.chemkit_next.json

# 4. doctor（あれば）
python tools/codex_prompt.py doctor || true
```

## 3) Autopilotで実行
```bash
./tools/autopilot.sh 30
```
既存の優先度ルールに従って、次の todo を処理します。

## 4) 置き換えが必要なファイルは？
このパックは **原則置き換え不要**です。
もし `tools/queue_merge_append.py` などが同名で既に存在する場合は、名前を変えて配置してください。
