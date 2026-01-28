# Memo

## Environment setup (Windows commands)

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt

# PyTorch + PyG (CPU wheels)
python -m pip install "torch==2.7.1"
python -m pip install torch-geometric
python -m pip install --force-reinstall torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.1+cpu.html

# RDKit-compatible numpy pin
python -m pip install --force-reinstall "numpy==1.26.4"

# torchmd-net (optional)
python -m pip install -e work/torchmd-net --no-build-isolation
```

## LJ training command (organic)

```powershell
python scripts/train.py --config configs/gnn/train_mpnn_lj_dataset_organic.yaml
```
## Training commands (LJ organic)

```bash
python3 scripts/train.py --config configs/gnn/train_mpnn_lj_dataset_organic.yaml
python3 scripts/train.py --config configs/gnn/train_mpnn_lj_dataset_organic.yaml2
```
