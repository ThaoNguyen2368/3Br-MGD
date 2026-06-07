# 3Br-MGD Implementation

This is the source code repository for the 3Br-MGD model and related baseline comparisons.

## Highlights
- **Unified Evaluation Interface** across multiple few-shot learning baselines (FS-GCvTR, FS-GNNTR, GCN, GIN, GraphSAGE, AttFPGNN) with shared utilities for logging, evaluation, and checkpointing.
- **Dataset-ready Directory Layout** for benchmark molecular property prediction datasets (`tox21` and `sider`).
- **Comprehensive Benchmarking** through an automated script (`run_all_baselines.py`) that seamlessly runs training and testing protocols.
- **Metrics**: AUROC evaluated under standardized N-way K-shot settings (e.g., 5-shot, 10-shot).

## Repository Layout
```text
├── 3Br_MGD/                # Core model definitions and variations (Br_MGD, Branch, ChemBERTa)
│   ├── Br_MGD/             # 3Br-MGD training and evaluation scripts
│   └── Data/               # Preprocessed splits and graphs for datasets (tox21, sider)
├── baselines/              # Implementation of all baseline models
│   ├── run_all_baselines.py# Master script to execute training/testing across all baselines
│   └── episode_manager.py  # Utility for consistent few-shot episode generation
├── checkpoints/            # Saved model checkpoints across experiments
├── results/                # Output metrics, performance summaries, and logs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation (this file)
```

## Prerequisites
- **Python 3.8+** (Tested with Conda environments)
- **CUDA-capable GPU** (Strongly recommended for Graph Neural Networks training)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Datasets
The repository expects preprocessed datasets under `3Br_MGD/Data/<dataset-name>/processed/` (e.g., `tox21` and `sider`). 
These folders contain:
- `dataset_info.json`: Metadata defining classes and structures.
- `meta_train/` & `meta_test/`: PyTorch Geometric (`.pt`) graph data for individual tasks.

*(Note: Due to GitHub file size limits, large `.pt` and `.json` data files are not included here. The Tox21 and SIDER datasets, as well as the test episodes (`episodes_seed42_tox21.json`, `episodes_seed42_sider.json`), can be downloaded from [this Google Drive link](https://drive.google.com/drive/u/0/folders/1uDY_SWy8gvzAziBkcnJMuHFzJI8IRDrr)).*

## Running Experiments

### 1. Baselines
All training and testing for baselines are managed by a centralized runner script: `run_all_baselines.py`.

**Train all baselines (e.g., on Tox21):**
```bash
python baselines/run_all_baselines.py --dataset tox21 --shots 5 10 --mode train
```

**Test all baselines:**
*(Make sure episodes are generated first using `episode_manager.py` if testing in strict predefined episodes)*
```bash
python baselines/run_all_baselines.py --dataset tox21 --shots 5 10 --mode test
```

**Run everything at once (Train + Test):**
```bash
python baselines/run_all_baselines.py --dataset tox21 --shots 5 10 --mode all
```

### 2. Main Model (3Br-MGD)
The core 3Br-MGD model is run directly from the `3Br_MGD/Br_MGD/` directory.

**Training 3Br-MGD on Tox21:**
```bash
python 3Br_MGD/Br_MGD/BrMGD_train.py \
    --data_dir 3Br_MGD/Data/tox21/processed \
    --output_dir checkpoints/BrMGD \
    --dataset tox21 \
    --shots 5 10 \
    --max_epochs 1000
```

**Testing 3Br-MGD on Tox21:**
```bash
python 3Br_MGD/Br_MGD/BrMGD_eval.py \
    --data_dir 3Br_MGD/Data/tox21/processed \
    --ckpt_dir checkpoints/BrMGD \
    --output_dir results/tox21 \
    --dataset tox21 \
    --shots 5 10 \
    --episodes_file baselines/episodes_seed42_tox21.json
```

Logs, checkpoints, and metrics CSVs are automatically saved under `checkpoints/` and `results/`.