# S64: Pretraining & Transfer

## 目的
- upstream pretrain を downstream に確実に引き継ぎ、比較可能にする

## 方式
- load_pretrained_weights（strict=False）
- freeze encoder / head only（linear probe）を config で切替

## Artifact
- upstream run_id を meta.upstream_artifacts に必ず残す
