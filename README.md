# 3Br-MGD: few-shot toxicity prediction with a three-branch deep encoder and meta-learning framework

This repository contains the official implementation of the paper:

**"3Br-MGD: few-shot toxicity prediction with a three-branch deep encoder and meta-learning framework"**

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Dataset](#dataset)
5. [Pre-generated Evaluation Episodes](#pre-generated-evaluation-episodes)
6. [Usage Guide](#usage-guide)
   - [Quick Start](#quick-start)
   - [Data Preparation](#data-preparation)
   - [Training](#training)
   - [Evaluation](#evaluation)

---

## 1. System Requirements

- **Python:** 3.11
- **PyTorch:** 2.3.0
- **CUDA:** 12.1
- **PyTorch Geometric:** 2.7.0
- **RDKit:** 2025.9.5
- **Other core libraries:** `numpy`, `pandas`, `scikit-learn`, `transformers`

## 2. Repository Structure

```text
├── 3Br_MGD/                 # Core model definitions and variations (3Br-MGD)
│   ├── Br_MGD/              # 3Br-MGD training and evaluation scripts
│   └── Data/                # Preprocessed splits and graphs for datasets (tox21, sider)
│
├── baselines/               # Implementation of all baseline models
│   ├── fsgcvtr/             # FS-GCvTR baseline
│   ├── fsgnntr/             # FS-GNNTR baseline
│   ├── attfpgnn/            # AttFPGNN baseline 
│   ├── gcn/                 # GCN baseline
│   ├── gin/                 # GIN baseline
│   ├── graphsage/           # GraphSAGE baseline
│   └── pre-trained/         # Pre-trained baseline weights
│
├── checkpoints/             # Directory for saving trained model weights
├── results/                 # Directory for storing evaluation metrics (JSON format)
└── README.md              
```

## 3. Environment Setup

We recommend using `conda` to manage the environment:

```bash
conda create -n 3Br_MGD python=3.11 -y
conda activate 3Br_MGD
```

Install the required dependencies directly via `pip`:

```bash
pip install -r requirements.txt
```

## 4. Dataset

The repository currently supports the **Tox21** and **SIDER** datasets for few-shot molecular property prediction.

The Tox21 and SIDER datasets are downloaded from the repository Data (chem_dataset.zip) from Hu et al. (2020). 
The data should be placed and processed inside `3Br_MGD/Data/`. You can run the `3Br_MGD/Br_MGD/data.py` script to perform the preprocessing steps automatically.

### Data Preparation
For dataset preprocessing, run:
```bash
python 3Br_MGD/Br_MGD/data.py
```
## 5. Pre-generated Evaluation Episodes

To ensure fair and reproducible evaluation, all methods are tested on the same pre-generated evaluation episodes. 

Files:
- `baselines/episodes_seed42_tox21.json`
- `baselines/episodes_seed42_sider.json`

These files contain the exact support/query task splits used in the reported experiments. The files can be accessed or cited via Zenodo:
**Download/DOI:** [https://doi.org/10.5281/zenodo.20590676](https://doi.org/10.5281/zenodo.20590676)

## 6. Usage Guide
### Training
**Training 3Br-MGD:**
```bash
python 3Br_MGD/Br_MGD/BrMGD_train.py --data_dir 3Br_MGD/Data/tox21/processed --dataset tox21 --shots 5 10 --max_epochs 1000 --patience 100
```

**Training Baselines:**
```bash
python baselines/run_all_baselines.py --dataset tox21 --shots 5 10 --mode train --max_epochs 1000 --patience 100 --baselines fsgnntr
```

**Training Branches:**
```bash
python 3Br_MGD/Branch/branch_Train.py --data_dir 3Br_MGD/Data/tox21/processed --output_dir checkpoints --dataset tox21 --variants all 
```

### Evaluation
**Testing 3Br-MGD:**
```bash
python 3Br_MGD/Br_MGD/eval_model.py --data_dir 3Br_MGD/Data/tox21/processed --dataset tox21 --shots 5 10 --episodes_file baselines/episodes_seed42_tox21.json
```

**Testing Branches:**
```bash
python 3Br_MGD/Br_MGD/eval_model.py --data_dir 3Br_MGD/Data/sider/processed --checkpoint_dir /home/fit03/BrMGD/3Br-MGD/checkpoints --dataset sider --episodes_file baselines/episodes_seed42_tox21.json
```
To view the results for 3Br-MGD and its branches, check the outputs located in `results/mean-3BrMGD_<dataset>_<shot>shot.txt` and `results/ablation_<dataset>_summary.json`.

**Testing Baselines:**
```bash
python baselines/run_all_baselines.py --dataset tox21 --shots 5 10 --mode test --baselines fsgnntr
```
To view a results of baselines, check the outputs located in `checkpoints/results_tox21.json`

**Metrics Evaluated:** AUROC, AUPRC, F1-Score, Accuracy
