# training_nemdo

`training_nemdo` is a PyTorch-based framework for training **neural mesh-free differential operators (NeMDO)**.  
The framework supports training with PyTorch and PyTorch Geometric and a modular pipeline for data preparation, training, evaluation, and deployment.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Workflow](#workflow)
  - [Data Import and Preprocessing](#data-import-and-preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [TorchScript Export](#torchscript-export)
- [Requirements](#requirements)

---

## Features

- End-to-end training pipeline for neural mesh-free differential operators
- Modular workflow with separate scripts for data import, training, evaluation, and deployment
- Flexible dataset sizing via a `data_iteration` parameter
- Explicit train / validation / test split with consistent normalisation
- TorchScript export for direct use in numerical simulations

---

## Repository Structure

```text
training_nemdo/
├── main_import.py        # data import, normalisation, train/val/test split
├── main_train.py         # graph construction and model training
├── main_test.py          # evaluation of learned operator moments
├── main_torchscript.py   # TorchScript export for simulation use
├── models/               # neural network architectures
├── requirements.txt
└── README.md
```
---

## Installation

### 1. Clone the repository

Using SSH:

```bash
git clone git@github.com:nemdo/training_nemdo.git
cd training_nemdo
```

## Workflow
The training pipeline is organised into four explicit stages, each handled by a dedicated script.

### Data Import and Preprocessing

**Entry point:** `main_import.py`

This script:
- loads the raw dataset,
- applies normalisation,
- creates train / validation / test splits,
- controls the dataset size via the `data_iteration` variable  
  (higher iterations correspond to larger datasets).

Run:

```bash
python main_import.py
```

### Training

Run:
**Entry point:** `main_train`
Model architecture, training hyperparameters, and DDP settings are configured directly within this script or via the corresponding modules in `models/`.

#### Variable (growing) stencils

With `variable_stencil = True` in both `main_import.py` and `main_train_gpu.py`, every point starts with a stencil of
`initial_stencil` (5) neighbours and its own stencil grows during training:

- `main_import.py` stores the raw (un-normalised) neighbour offsets, up to `max_neighbours`. Each point's offsets are
  normalised on the fly by the largest distance in its *current* stencil (`VariableStencilGraph`).
- When the training loss stalls (`growth_patience` epochs without a `growth_rel_tol` relative improvement), the points
  with the worst per-point moment loss (above `growth_quantile`) gain `stencil_step` neighbours, up to `max_neighbours`
  (`functions/stencil_growth.py`). Points below `converged_tol` stop growing.
- The per-point sizes are saved in the checkpoints (`stencil_sizes`) and restored when resuming.
- Best-validation tracking restarts after each growth, since losses on different stencils are not comparable.

`main_test.py` still expects the fixed-size, pre-normalised data (`variable_stencil = False`).

### Evaluation

**Entry point:** `main_test`
This script evaluates the trained model on the test dataset, focusing on:
- moment consistency,
- qualitative results of predicted weights

Run:
```bash
python main_test.py
```

### TorchScript Export
**Entry point:** `main_torchscript.py`

This script converts the trained model into a TorchScript representation, enabling deployment in production codes.

Run:
```bash
python main_torchscript.py
```

## Data Availability

The dataset used in this work is hosted on Hugging Face:

https://huggingface.co/datasets/nemdo/nemdo-data

The repository contains the full dataset used for training, validation, and testing.
Within the dataset repository, the data are stored under the `dataset/` directory.

The dataset can be downloaded programmatically using the Hugging Face Hub API. For example:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nemdo/nemdo-data",
    repo_type="dataset",
    local_dir="data",
)
