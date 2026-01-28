# Environment Setup

This document describes a stable setup flow for this repo, including PyTorch Geometric (PyG) native extensions.

## Prerequisites

- Python 3.10
- pip
- macOS: Xcode Command Line Tools

## Base install (repo)

```
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## PyTorch + PyG (recommended)

PyG must be installed with its native extensions matching the **exact torch version**.

1) Install torch (CPU example):

```
python3 -m pip install "torch==2.7.1"
```

2) Install PyG core:

```
python3 -m pip install torch-geometric
```

3) Install PyG native extensions (must match torch):

```
python3 -m pip install --force-reinstall \
  torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.7.1+cpu.html
```

If you use a different torch version, change the URL accordingly.

## torchmd-net (optional)

If you need `work/torchmd-net`:

```
MACOSX_DEPLOYMENT_TARGET=11.0 \
python3 -m pip install -e work/torchmd-net --no-build-isolation
```

This avoids a macOS SDK compatibility error during C++ extension build.

## Sanity checks

Run these before tests:

```
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch_geometric; print(torch_geometric.__version__)"
python3 -c "import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv; print('pyg extensions ok')"
python3 -c "import rdkit; print(rdkit.__version__)"
```

## Troubleshooting

- **`import torch_geometric` crashes (Signal 6)**
  - Usually means PyG native extensions do not match torch.
  - Reinstall extensions using the correct wheel URL (see above).

- **torchmd-net build fails on macOS**
  - Ensure `MACOSX_DEPLOYMENT_TARGET=11.0` is set during install.

- **numpy version jumps**
  - Some pip installs can upgrade numpy; verify compatibility with RDKit.
  - If needed, pin numpy in your environment and reinstall dependent wheels.
