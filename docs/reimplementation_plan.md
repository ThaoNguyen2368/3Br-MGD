# Reimplementation Plan — FS-GNNTR Baselines under 3Br-MGD Protocol

> **Status**: Design Proposal — Awaiting User Approval  
> **Date**: 2026-05-30  
> **Prerequisite**: Phase 0 analysis (`docs/fs_gnntr_integration_plan.md`) approved

---

## Overview

This document describes the full integration plan for running FS-GNNTR, GCN, GIN, and GraphSAGE
baselines under the exact same experimental protocol as 3Br-MGD, enabling fair, reviewer-proof
comparisons.

---

## Which Components Will Remain Unchanged

### From FS-GNNTR Repository (untouched)

| Component | File | Reason |
|-----------|------|--------|
| GINConv message-passing | `gnn_models.py` | Original architecture |
| GCNConv message-passing | `gnn_models.py` | Original architecture |
| GraphSAGEConv message-passing | `gnn_models.py` | Original architecture |
| GATConv message-passing | `gnn_models.py` | Original architecture |
| GNN multi-layer backbone | `gnn_models.py` | Original architecture |
| GNN_prediction (pooling head) | `transformer.py` | Original architecture |
| TR (Vision Transformer) | `transformer.py` | Original architecture |
| AttentionLayer, FeedForwardLayer | `transformer.py` | Original architecture |
| MAML update logic | `gnntr_train.py` | Original training algorithm |
| BCEWithLogitsLoss objective | `gnntr_train.py` | Original loss function |
| Pre-trained weight loading | `transformer.py` | Required for fair comparison |
| Pre-trained `.pth` files | `pre-trained/` | Mandatory for reproducibility |

### From 3Br-MGD (untouched)

| Component | File | Reason |
|-----------|------|--------|
| Task split definitions | `data.py` (TOX21_SPLITS, SIDER_SPLITS) | Authoritative split |
| `preprocess()` + `load_all_splits()` | `data.py` | Authoritative data pipeline |
| `atom_features()` / `bond_features()` | `data.py` | Authoritative feature encoding |
| `smiles_to_graph()` | `data.py` | Authoritative graph construction |
| `evaluate_meta_task()` | `BrMGD_eval.py` | Shared metric implementation |
| `roc_auc_score` usage | `BrMGD_eval.py` | Shared metric function |
| `collate_batch()` | `BrMGD_eval.py` | Shared batch collation |
| TripleEncoder / EnhancedProtoNet | `BrMGD_model.py` | 3Br-MGD architecture |

---

## Which Components Will Be Replaced (per baseline)

| Component | Original | Replacement |
|-----------|----------|-------------|
| Task split | Numeric index (task_1..task_12) | Named tasks from `TOX21_SPLITS` / `SIDER_SPLITS` |
| Data loader | `MoleculeDataset` (per-task pickle directories) | `load_all_splits()` from 3Br-MGD `data.py` |
| Support/query sampler | `random_sampler()` (unseeded, hardcoded tables) | `create_meta_task()` from `BrMGD_train.py` |
| Episode generation | Internal to `GNNTR.meta_train()` | Shared `EpisodeManager` loading `episodes_seed42.json` |
| ROC-AUC evaluation | `roc_accuracy()` inside `gnntr_eval.py` | `evaluate_meta_task()` from `BrMGD_eval.py` |
| Result file format | Custom `results-exp/` txt files | Shared `results.json` schema |
| Checkpoint format | `{'state_dict': ..., 'optimizer': ..., 'epoch': ...}` | `{'model_state': ..., 'results': ..., 'config': ...}` |

---

## Shared Evaluation Protocol

All methods — including 3Br-MGD — will be evaluated identically:

### Task Split (Shared)

```python
# Tox21
meta_train = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
              'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5']

meta_test  = ['SR-HSE', 'SR-MMP', 'SR-p53']

# SIDER
meta_train = [21 named tasks...]
meta_test  = ['Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions',
              'Ear and labyrinth disorders', 'Cardiac disorders',
              'Nervous system disorders', 'Injury, poisoning and procedural complications']
```

Source: `3Br_MGD/Br_MGD/data.py` — `TOX21_SPLITS` / `SIDER_SPLITS` (unchanged).

### Seed Control (Shared)

```python
# baselines/seed_utils.py
def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Applied before:
- Preprocessing (data.py)
- Episode generation
- Model initialization
- Training loop
- Testing loop

### Episodes (Shared)

All models use identical pre-generated episodes:

```json
// baselines/episodes_seed42.json
{
  "dataset": "tox21",
  "seed": 42,
  "shots": [5, 10],
  "meta_test_episodes": {
    "5-shot": [
      {
        "task": "SR-HSE",
        "support_pos_smiles": ["..."],
        "support_neg_smiles": ["..."],
        "query_pos_smiles": ["..."],
        "query_neg_smiles": ["..."]
      },
      ...
    ],
    "10-shot": [...]
  }
}
```

Episode generation logic:
- `set_seed(42)` first
- For each test episode (30 episodes per task), call `create_meta_task(task_data, K_shot, Q_query=None, train=False)`
- Store SMILES strings (not tensors) so any model can reconstruct needed format

### Evaluation Metric (Shared)

```python
# Shared function from BrMGD_eval.py
from BrMGD_eval import evaluate_meta_task  # acc, f1, auroc

# Primary metric: ROC-AUC
# Report: mean ± std over 30 test episodes per task
```

All baselines call the same `roc_auc_score(y_true, y_pred_proba)` with identical query sets.

---

## Phase 2 — Reproducibility Framework

### Directory Structure

```
baselines/
    seed_utils.py           # set_seed(seed=42)
    episode_manager.py      # generate_episodes(), load_episodes(), EpisodeManager
    graph_adapter.py        # adapt_graph_to_fsgnntr() — format converter
    episodes_seed42_tox21.json
    episodes_seed42_sider.json
    dataloaders/
        __init__.py
        fsgnntr_loader.py   # load_task_as_fsgnntr() — returns long-index graphs
    fsgnntr/
        train.py
        test.py
        config.py
        adapter.py
    gcn/
        train.py
        test.py
        config.py
        adapter.py
    gin/
        train.py
        test.py
        config.py
        adapter.py
    graphsage/
        train.py
        test.py
        config.py
        adapter.py
```

### `seed_utils.py`

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### `episode_manager.py`

```python
class EpisodeManager:
    """
    Generates or loads shared test episodes.
    All models load from the same JSON file.
    No model generates its own episodes.
    """
    def generate_and_save(data_dir, dataset, K_shots, n_episodes, seed, out_path):
        set_seed(seed)
        meta_train, meta_test = load_all_splits(data_dir)
        episodes = {}
        for K_shot in K_shots:
            episodes[f"{K_shot}-shot"] = []
            for ep in range(n_episodes):
                for task_name, task_data in meta_test.items():
                    support, query = create_meta_task(task_data, K_shot, Q_query=None, train=False)
                    episodes[f"{K_shot}-shot"].append({
                        "task": task_name,
                        "episode": ep,
                        "support_smiles": [s['smiles'] for s in support],
                        "support_labels": [s['label'] for s in support],
                        "query_smiles": [s['smiles'] for s in query],
                        "query_labels": [s['label'] for s in query],
                    })
        with open(out_path, 'w') as f:
            json.dump(episodes, f, indent=2)

    def load(path):
        with open(path) as f:
            return json.load(f)
```

### `graph_adapter.py`

```python
# Converts 3Br-MGD float one-hot graphs -> FS-GNNTR long-index graphs
# Does NOT modify any model code

ATOM_SYMBOLS_3BR = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca',
                    'Fe','As','Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag',
                    'Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni',
                    'Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']

# Atomic numbers for each symbol (for FS-GNNTR allowable_features mapping)
SYMBOL_TO_ATOMIC_NUM = {
    'C':6,'N':7,'O':8,'S':16,'F':9,'Si':14,'P':15,'Cl':17,'Br':35,'Mg':12,
    'Na':11,'Ca':20,'Fe':26,'As':33,'Al':13,'I':53,'B':5,'V':23,'K':19,
    'Tl':81,'Yb':70,'Sb':51,'Sn':50,'Ag':47,'Pd':46,'Co':27,'Se':34,
    'Ti':22,'Zn':30,'H':1,'Li':3,'Ge':32,'Cu':29,'Au':79,'Ni':28,'Cd':48,
    'In':49,'Mn':25,'Zr':40,'Cr':24,'Pt':78,'Hg':80,'Pb':82,'Unknown':1
}

POSSIBLE_ATOMIC_NUM_LIST = list(range(1, 119))

def adapt_graph_to_fsgnntr(graph_3br):
    """
    Convert 3Br-MGD Data (float one-hot) to FS-GNNTR Data (long index).
    """
    N = graph_3br.x.shape[0]
    atom_oh = graph_3br.x[:, :44].argmax(dim=1)  # [N] indices into ATOM_SYMBOLS_3BR

    atom_num_idx = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        symbol = ATOM_SYMBOLS_3BR[atom_oh[i].item()]
        atomic_num = SYMBOL_TO_ATOMIC_NUM.get(symbol, 1)
        fsgnntr_idx = POSSIBLE_ATOMIC_NUM_LIST.index(atomic_num) if atomic_num in POSSIBLE_ATOMIC_NUM_LIST else 0
        atom_num_idx[i] = fsgnntr_idx
    chirality_idx = torch.zeros(N, dtype=torch.long)  # default: CHI_UNSPECIFIED
    x_new = torch.stack([atom_num_idx, chirality_idx], dim=1)  # [N, 2] long

    E = graph_3br.edge_attr.shape[0]
    bond_oh = graph_3br.edge_attr[:, :4].argmax(dim=1)   # [E] 0=SINGLE,1=DOUBLE,2=TRIPLE,3=AROMATIC
    bond_dir_idx = torch.zeros(E, dtype=torch.long)       # default: NONE
    edge_attr_new = torch.stack([bond_oh.long(), bond_dir_idx], dim=1)  # [E, 2] long

    return Data(
        x=x_new,
        edge_index=graph_3br.edge_index.clone(),
        edge_attr=edge_attr_new,
    )
```

---

## Phase 3 — Baseline Integration

### Per-Baseline Directory Contents

Each `baselines/<model>/` directory contains:

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters (gnn_type, emb_size, layers, lr, K_shot, etc.) |
| `adapter.py` | Input format conversion (imports from `baselines/graph_adapter.py`) |
| `train.py` | Meta-training using 3Br-MGD `meta_train` tasks + shared sampler |
| `test.py` | Evaluation using 3Br-MGD `meta_test` tasks + shared `episodes_seed42.json` |

### FS-GNNTR (`baselines/fsgnntr/`)

**config.py:**
```python
DATASET = "tox21"
GNN_TYPE = "gin"        # "gin", "gcn", "graphsage"
EMB_SIZE = 300
GRAPH_LAYERS = 5
N_SUPPORT = 10          # K-shot
N_QUERY = 256
BATCH_SIZE = 10         # MUST be 10 (hardcoded in TR)
LR_GNN = 0.001
LR_TR = 1e-5
LR_UPDATE = 0.5         # MAML inner update rate
K_TRAIN = 10            # inner steps (training)
K_TEST = 20             # inner steps (testing)
PRETRAINED = "FS-GNNTR_repo/FS-GNNTR/pre-trained/supervised_contextpred.pth"
SEED = 42
POS_WEIGHT = 25.0       # for tox21; 1.0 for sider
```

**train.py (pseudocode):**
```python
set_seed(42)
meta_train, _ = load_all_splits(data_dir)
# Convert all graphs in meta_train to FS-GNNTR format (adapter)
# Use create_meta_task() for episode sampling
# Use GNNTR.meta_train() logic adapted for named tasks
# Save: checkpoints/fsgnntr/best_model.pt, last_model.pt
```

**test.py (pseudocode):**
```python
set_seed(42)
episodes = EpisodeManager.load("baselines/episodes_seed42_tox21.json")
# For each episode in episodes["10-shot"]:
#   Reconstruct graphs from SMILES using FS-GNNTR adapter
#   Run MAML adaptation on support set
#   Evaluate on query set using roc_auc_score
# Save: checkpoints/fsgnntr/results.json
```

### GCN (`baselines/gcn/`)

Same structure as FS-GNNTR but:
- `baseline=1` (no Transformer)
- `gnn_type="gcn"`
- Pretrained: `gcn_supervised_contextpred.pth`

### GIN (`baselines/gin/`)

Same structure but:
- `baseline=1`
- `gnn_type="gin"`
- Pretrained: `supervised_contextpred.pth`

### GraphSAGE (`baselines/graphsage/`)

Same structure but:
- `baseline=1`
- `gnn_type="graphsage"`
- Pretrained: `graphsage_supervised_contextpred.pth`

---

## Phase 4 — Training Protocol

All baselines train using `meta_train` tasks ONLY.

**Episode sampling during training:**
```python
set_seed(42)
meta_train, _ = load_all_splits(data_dir)
# At each training step:
task_name = random.choice(list(meta_train.keys()))
support, query = create_meta_task(meta_train[task_name], K_shot, Q_query, train=True)
# -> adapt_graph_to_fsgnntr(sample['graph']) for each sample
```

**Training continues for `max_epochs=200` with `patience=20` early stopping (same as 3Br-MGD).**

---

## Phase 5 — Testing Protocol

All baselines test using `meta_test` tasks ONLY, with identical shared episodes.

**Episode loading:**
```python
episodes = EpisodeManager.load("baselines/episodes_seed42_tox21.json")
# episodes["10-shot"] = list of {task, episode, support_smiles, query_smiles, ...}
```

**For each episode:**
1. Reconstruct samples from SMILES using each baseline's adapter
2. Run model-specific adaptation on support (MAML k_test steps)
3. Evaluate on query using `roc_auc_score(query_labels, model_proba)`
4. Append auroc to per-task list

**30 test episodes per task (same as 3Br-MGD eval_model.py default).**

---

## Phase 6 — Metrics

Primary metric: **ROC-AUC**

```python
# Identical to BrMGD_eval.py evaluate_meta_task()
from sklearn.metrics import roc_auc_score

auroc = roc_auc_score(query_y_true, query_proba)
```

Report per task:
```
task_name: auroc_mean ± auroc_std  (over 30 episodes)
```

Report overall:
```
mean_auroc ± std_auroc  (mean of per-task means)
```

---

## Phase 7 — Outputs

### Checkpoint Structure

```
checkpoints/
    fsgnntr/
        best_model_gnn.pt        # GNN state dict (best val)
        best_model_tr.pt         # Transformer state dict (best val)
        last_model_gnn.pt
        last_model_tr.pt
        results.json
    gcn/
        best_model.pt
        last_model.pt
        results.json
    gin/
        best_model.pt
        last_model.pt
        results.json
    graphsage/
        best_model.pt
        last_model.pt
        results.json
```

### results.json Schema

```json
{
  "model": "fsgnntr",
  "dataset": "tox21",
  "gnn_type": "gin",
  "shots": 10,
  "seed": 42,
  "meta_train_tasks": ["NR-AR", "NR-AR-LBD", ...],
  "meta_test_tasks": ["SR-HSE", "SR-MMP", "SR-p53"],
  "n_test_episodes": 30,
  "auc_mean": 0.0,
  "auc_std": 0.0,
  "per_task": {
    "SR-HSE": {"auc_mean": 0.0, "auc_std": 0.0, "raw_auc": [...]},
    "SR-MMP": {"auc_mean": 0.0, "auc_std": 0.0, "raw_auc": [...]},
    "SR-p53": {"auc_mean": 0.0, "auc_std": 0.0, "raw_auc": [...]}
  }
}
```

---

## Open Questions (Require User Decision Before Implementation)

> [!IMPORTANT]
> The following design decisions must be resolved before any code is written.

### Q1: Scope of Baselines

The FS-GNNTR repository contains only GCN, GIN, GraphSAGE, GAT, and FS-GNNTR.
The task specification also mentions: **FS-GCvTR, Meta-MGNN, MAML, Seq2Seq, EGNN**.

These are NOT in the cloned repository.

**Options:**
- A) Limit integration to what is in the repo: FS-GNNTR, GCN, GIN, GraphSAGE, (GAT optional)
- B) Source additional baselines from separate repositories (Meta-MGNN, EGNN, Seq2Seq, MAML)
- C) Implement missing baselines from scratch based on original papers

**Decision needed**: Which additional baselines are in scope?
A
### Q2: Pre-trained Weights Dependency

All FS-GNNTR baselines require pre-trained GNN weights (`supervised_contextpred.pth`, etc.).
These are included in the repo (≈7MB each).

Without pre-training, the baselines would not be faithful to the original paper.

**Question**: Should baselines be run WITH pre-trained weights (faithful to paper) or WITHOUT (pure from-scratch, which would be an unfair comparison since the paper uses pre-training)?

**Recommendation**: Use pre-trained weights. Document this explicitly in the paper as a controlled variable.
=> Use pre-trained weights
### Q3: TR Batch Size Constraint

The FS-GNNTR Vision Transformer has `emb.reshape(10, 1, 300, 1)` hardcoded in `transformer.py:105`.
This means every batch must be exactly size 10.

**Options:**
- A) Pad/truncate support and query sets to multiples of 10 (may distort results)
- B) Modify only the `reshape` line to be dynamic (minimal, justified change)
- C) Accept the batch_size=10 constraint and design episodes accordingly

**Note**: Option B changes one line in the original code but does not modify the architecture.
B
### Q4: Comparison of Training Paradigms

3Br-MGD uses **Prototypical Networks** (metric learning, no gradient update at test time).  
FS-GNNTR uses **MAML** (gradient-based meta-learning, k_test=20 gradient steps at test time).

These are fundamentally different meta-learning paradigms. A reviewer might argue the comparison is unfair regardless of shared episodes.

**Recommendation**: Keep each model's training paradigm intact (faithful comparison). Explicitly acknowledge in the paper that different meta-learning strategies are compared under controlled data conditions.
OK
### Q5: MAML Training Duration

FS-GNNTR's `train_model.py` runs for 2000 epochs. 3Br-MGD runs for 200 epochs with early stopping.

**Options:**
- A) Run baselines for their original duration (2000 epochs)
- B) Apply the same 200 epochs + patience=20 early stopping to all models
- C) Let each model run to convergence, then select best checkpoint

**Recommendation**: Option B — apply same stopping criterion to all. This is more reviewer-proof.
B
### Q6: Episode Pre-generation Strategy

Two options for shared episodes:
- A) Pre-generate episodes before any training and save to JSON (recommended for full reproducibility)
- B) Use a fixed seed at test time for all models (simpler but requires careful ordering)

**Recommendation**: Option A — pre-generate once from 3Br-MGD's data.py pipeline.
A
---

## Verification Plan

### Automated Tests
1. `python baselines/episode_manager.py --generate` — verify episode file created correctly
2. `python baselines/graph_adapter.py --test` — verify adapted graph has correct dtype and shape
3. `python baselines/gin/train.py --dry-run` — verify training loop runs without error
4. `python baselines/gin/test.py --episodes baselines/episodes_seed42_tox21.json` — verify metric output

### Manual Verification
1. Confirm `results.json` task names match `meta_test` task names from `data.py`
2. Confirm seed is set identically in 3Br-MGD `eval_model.py` and all baseline `test.py`
3. Confirm same 30 episodes are used by cross-checking SMILES in `episodes_seed42.json` against 3Br-MGD query outputs
4. Spot-check ROC-AUC for one task manually

---

*End of Reimplementation Plan — Awaiting user approval before any code is written.*
