# ChemKit Next Codex Playbook

このフォルダは「ChemKit Next」の拡張タスクを Codex/Autopilot で進める際の補助資料です。

## 典型の落とし穴
- ライブラリ依存（PyGのバージョン差、e3nn、torchmd-net）
- データセットの取得経路（PyG内蔵/公式配布/OGB）
- 単位・ターゲット名の揺れ（target registryで吸収）
- 3D座標生成（失敗・多コンフォーマ）
- マルチタスクの欠損ターゲット（mask_y必須）

## 自動化運用ルール
- 進められない場合は「blocked」ではなく、まず **depends_on** で順序を調整（順番待ち）
- 外部依存が必要で環境が無い場合のみ blocked + 解除タスク起票（unblocks）
