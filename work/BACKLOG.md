# バックログ（優先度付き）

このバックログは「何十回もの Codex 反復」を前提に、タスクを小さく切っています。

| Priority | Task ID | Title | Outcome |
|---|---:|---|---|
| P0 | 000 | devkit導入（docs/work/agentskills/codex） | ブレない開発の土台 |
| P0 | 010 | Hydra config の group 再整理 | 実験管理の統一 |
| P0 | 015 | Process単位のCLIエントリポイント整理 | 独立実行/将来ClearML化の土台 |
| P0 | 020 | Artifact 契約の実装 | 再現性・運用性 |
| P0 | 030 | FeaturePipeline 統一（train/infer skew排除） | 精度/運用の地雷除去 |
| P1 | 040 | Task（目的変数）抽象化（LJ以外を追加可能に） | 物性拡張 |
| P1 | 050 | データ収集（外部API）モジュール設計 | 収集機能拡張 |
| P1 | 060 | モデル追加テンプレ（例: GIN/GAT） | モデル拡張 |
| P2 | 070 | 事前学習埋め込み特徴量の導入 | 精度改善の選択肢 |
| P2 | 080 | 実験追跡（MLflow/W&Bなど任意） | 比較・再現 |
| P2 | 090 | CI（lint/test/smoke）整備 | 品質維持 |


> 注意: Process（処理単位）を増やした/分割した場合は docs/10_PROCESS_CATALOG.md を更新してください。
