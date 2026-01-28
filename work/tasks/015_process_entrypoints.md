# Task 015 (P0): Process単位のCLIエントリポイント整理（train/evaluate/predict/visualize 等）

## 目的
- 各処理を Hydra 管理の Process として **単独実行**できるようにする
- 将来 ClearML Task 化しやすい粒度にする
- 可視化等が学習スクリプトに混ざって肥大化するのを防ぐ

## スコープ
### In scope
- `scripts/` に process ごとの entrypoint を整理・追加（例: train/evaluate/predict/visualize/featurize）
- `configs/process/` を追加し、processごとの “入口config” を持つ
- 各Processが `config.yaml` と `meta.json` を出せる土台を作る（実装詳細は020に寄せてもOK）

### Out of scope
- ClearML SDK を入れて実装する（今回はやらない）
- 大規模なモデル改修（別タスク）

## Plan
1) 現状の scripts を棚卸し（train/predict/eval/viz が混在していないか）
2) “1 script = 1 process” の形に整理（必要なら新規 scripts/visualize.py を作る）
3) configs/process を導入し、各 process の defaults を定義
4) smoke test（最小データ）で train→evaluate→predict→visualize の実行動線を確認

## Acceptance Criteria
- [ ] 各Processが単独で実行できる
- [ ] process名が run dir / meta に残る（契約は docs/04）
- [ ] visualize が train/predict に混ざらず独立している（肥大化防止）
