# Supervised Pretrain Recipe (QM9 -> LJ)

## Goal
Minimal supervised pretrain recipe using QM9 multitask pretrain and LJ epsilon finetune.

## Configs
- Pretrain (QM9 multitask): `configs/gnn/train_qm9_pretrain.yaml`
- Finetune (LJ epsilon): `configs/gnn/train_lj_finetune_qm9.yaml`
- Benchmark report (baseline vs finetune): `configs/benchmark_suite_lj_pretrain.yaml`

## Prereqs
Run from the repo root. Ensure datasets are built:
```bash
python scripts/build_dataset.py --config configs/dataset_qm9.yaml
python scripts/build_dataset.py --config configs/dataset.yaml
```

## Run
1) Pretrain on QM9 (gap/homo/lumo multitask).
```bash
python scripts/train.py --config configs/gnn/train_qm9_pretrain.yaml
```
Output: `runs/train/gnn/qm9_pretrain_mpnn_v1`

2) Finetune on LJ epsilon using the pretrained weights.
```bash
python scripts/train.py --config configs/gnn/train_lj_finetune_qm9.yaml
```
This uses `configs/pretrain/qm9_supervised.yaml` which points at
`runs/train/gnn/qm9_pretrain_mpnn_v1`. If you change the pretrain exp name
or run dir, update `pretrain.ckpt_path` accordingly.

3) Compare baseline vs finetune (metrics + convergence artifacts).
```bash
python scripts/benchmark_suite.py --config configs/benchmark_suite_lj_pretrain.yaml
```
Report: `runs/benchmark_suite/lj_pretrain_transfer/benchmark_report.md`

## Notes
- Downstream `meta.json` captures `upstream_artifacts` plus `pretrain_*` fields.
- Train runs save `plots/learning_curve.png` (GNN) or `learning_curve_val.png` (FP);
  the benchmark report links to these when present.
- Optional PCQM4Mv2 pretrain requires the dataset adapter (Task 107) and is not included here.
