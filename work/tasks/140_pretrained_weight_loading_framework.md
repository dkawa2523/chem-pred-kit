# Task 140 (P0): Pretrained Weight Loading Framework

## 目的
事前学習済み重みを downstream に安全に適用できる共通機構を用意する。

## In scope
- config: `pretrain.ckpt_path`, `pretrain.mode`（finetune/linear_probe/partial_freeze）
- strict=False でロードし、head形状違いを許容
- freeze篙動の統一（optimizerのparam group管理）
- meta に upstream run_id / ckpt hash を記録

## Acceptance Criteria
- [x] 既存モデル（GCN等）でも「重み無しなら従来通り」
- [x] 3D/Transformerでも同一インターフェースでロードできる

## Verification
```sh
python - <<'PY'
from src.common.pretrain import build_pretrain_metadata
assert build_pretrain_metadata({}) is None
print("pretrain import ok")
PY
```
