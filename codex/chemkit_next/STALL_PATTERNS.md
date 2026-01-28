# Autopilot が止まりやすい出力パターン（ChemKit Next）

Codexが以下を出しがち：
- 「確認してください」
- 「blockedなので未実装」
- 「書き込み不可」
- 「Change Plan None」

対策：
- Autopilot mode では質問禁止・必ず前進（実装 or blocked+解除タスク）
- optional dependency は「必要なら明示して blocked+解除タスク」
