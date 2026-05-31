# FS-GNNTR Integration Plan — Phase 0 Analysis

> **Status**: Analysis Only — No Code Modified  
> **Date**: 2026-05-30  
> **Analyst**: Automated Repository Inspection

---

## 0.1 Repository Structure

### FS-GNNTR Repository (`FS-GNNTR_repo/FS-GNNTR/`)

#### Models Implemented

| File | Models / Classes |
|------|-----------------|
| `gnn_models.py` | `GINConv`, `GCNConv`, `GATConv`, `GraphSAGEConv`, `GNN` (multi-type backbone) |
| `transformer.py` | `GNN_prediction` (GNN head), `TR` (Vision Transformer), `Transformer`, `AttentionLayer`, `FeedForwardLayer` |
| `gnntr_train.py` | `GNNTR` class — wraps GNN + Transformer; implements MAML-style meta-training |
| `gnntr_eval.py` | `GNNTR_eval` class — evaluation mirror of `GNNTR` |

The single `GNN` backbone with `gnn_type` parameter implements all four graph architectures:
- **GIN** (`gnn_type="gin"`) — uses `GINConv`
- **GCN** (`gnn_type="gcn"`) — uses `GCNConv`
- **GAT** (`gnn_type="gat"`) — uses `GATConv`
- **GraphSAGE** (`gnn_type="graphsage"`) — uses `GraphSAGEConv`

The **FS-GNNTR** model = `GNN_prediction` (any GNN backbone) + `TR` (Vision Transformer), combined with MAML-style meta-learning.

The **baseline** mode (`baseline=1`) = `GNN_prediction` alone (no Transformer), also with MAML-style meta-learning. This is used for GCN, GIN, GraphSAGE standalone baselines.

> **Important Note**: There is NO separate "FS-GCvTR", "Meta-MGNN", "MAML", "Seq2Seq", or "EGNN" code in this repository. The repo implements only: FS-GNNTR (GNN+TR), and GNN-only baselines (GCN / GIN / GraphSAGE / GAT). The `baseline.txt` file in the 3Br-MGD project contains pasted copies of FS-GNNConv files — additional baseline code is not in the cloned repo.

#### Training Scripts
| Script | Purpose |
|--------|---------|
| `train_model.py` | Entry-point: instantiates `GNNTR`, loops 2000 epochs calling `meta_train()` + `meta_test()` |
| `gnntr_train.py` | `GNNTR.meta_train()` — MAML inner/outer loops over all train tasks; `GNNTR.meta_test()` — adaptation + evaluation on test tasks |

#### Evaluation Scripts
| Script | Purpose |
|--------|---------|
| `eval_model.py` | Entry-point: instantiates `GNNTR_eval`, loops 30 episodes calling `meta_evaluate()` |
| `gnntr_eval.py` | `GNNTR_eval.meta_evaluate()` — adaptation on support + ROC-AUC on query |

#### Preprocessing Scripts
| Script | Function |
|--------|----------|
| `data.py` | `split_into_directories()` — reads raw CSV, splits per task into pickle directories |
| `data.py` | `MoleculeDataset` (PyG `InMemoryDataset`) — loads pickled SMILES, converts to graphs using `mol_to_graph_data_obj_simple()` |
| `data.py` | `dataset()` — builds `MoleculeDataset` objects for all tasks |

#### Samplers
| Function | Location |
|----------|----------|
| `random_sampler(D, d, t, k, n, train)` | `data.py` — samples K support + N query from a task dataset using hardcoded class-size lookup tables |
| `sample_train(n_tasks, data, batch_size, n_support, n_query)` | `gnntr_train.py` — wraps `random_sampler` for all training tasks |
| `sample_test(tasks, test_task, data, batch_size, n_support, n_query)` | `gnntr_train.py`, `gnntr_eval.py` — wraps `random_sampler` for one test task |

#### Dataset Loaders
| Function | Location |
|----------|----------|
| `MoleculeDataset` | `data.py` — PyG InMemoryDataset; loads from per-task pickle directories |
| `_load_tox21_dataset()` | `data.py` — loads binary pickle for a single tox21 task |
| `_load_sider_dataset()` | `data.py` — loads binary pickle for a single sider task |
| `_load_muv_dataset()` | `data.py` — loads binary pickle for a single muv task |

#### Utility Scripts
| File | Purpose |
|------|---------|
| `utils/boxplots.py` | Result visualization only |
| `utils/significance.py` | Statistical significance tests |

#### Pre-trained Models (in `pre-trained/`)
18 pre-trained GNN weights from Hu et al. (2020). Used via `GNN_prediction.from_pretrained()`. Standard: `supervised_contextpred.pth` (GIN), `gcn_supervised_contextpred.pth`, `graphsage_supervised_contextpred.pth`, `gat_supervised_contextpred.pth`.

---

## 0.2 Baseline Inventory

### Model: FS-GNNTR

```
Model: FS-GNNTR

Location:
    gnntr_train.py   (GNNTR class, meta_train / meta_test)
    gnntr_eval.py    (GNNTR_eval class, meta_evaluate)
    transformer.py   (GNN_prediction + TR classes)
    gnn_models.py    (GNN backbone)

Input:
    graph only (molecular graph)

Uses:
    x          : [N_atoms, 2]  dtype=torch.long  (atom_type_idx, chirality_idx)
    edge_index : [2, N_edges]  dtype=torch.long
    edge_attr  : [N_edges, 2]  dtype=torch.long  (bond_type_idx, bond_dir_idx)
    batch      : [N_atoms]     dtype=torch.long

Architecture:
    GNN_prediction (5-layer GNN + mean pooling -> [B, 300])
    TR (Vision Transformer patch-embed on 300-dim vector -> [B, 1] logit + [B, 128] emb)
    CRITICAL: batch_size must be exactly 10 (hardcoded reshape in TR.forward)

Training:
    MAML-style meta-learning
    Inner loop: GNN params updated on support set (BCEWithLogitsLoss, pos_weight=25 for tox21)
    Outer loop: Transformer params updated on query set
    k_train=10 inner steps, k_test=20 test adaptation steps
    Optimizer: Adam (GNN lr=0.001, Transformer lr=1e-5)
    Requires pretrained GNN weights

Evaluation:
    ROC-AUC (primary)
    F1, Precision, Sensitivity, Specificity, Accuracy, Balanced Accuracy
```

### Model: GCN (FS-GNNTR baseline mode)

```
Model: GCN

Location:
    gnntr_train.py   (GNNTR class with baseline=1, gnn_type="gcn")
    gnn_models.py    (GCNConv, GNN)
    transformer.py   (GNN_prediction with gcn backbone)

Input:
    graph only

Uses:
    x          : [N_atoms, 2]  dtype=torch.long
    edge_index : [2, N_edges]  dtype=torch.long
    edge_attr  : [N_edges, 2]  dtype=torch.long
    batch      : [N_atoms]     dtype=torch.long

Architecture:
    5-layer GCNConv with edge embedding lookup + batch norm
    Mean pooling -> [B, 300] -> Linear -> [B, 1] logit
    Requires pretrained GNN weights (gcn_supervised_contextpred.pth)

Training:
    MAML inner loop on support set (BCEWithLogitsLoss, no pos_weight in baseline=1)
    k_train=10, k_test=20

Evaluation:
    ROC-AUC
```

### Model: GIN (FS-GNNTR baseline mode)

```
Model: GIN

Location:
    gnntr_train.py   (GNNTR class with baseline=1, gnn_type="gin")
    gnn_models.py    (GINConv, GNN)
    transformer.py   (GNN_prediction with gin backbone)

Input:
    graph only

Uses:
    x          : [N_atoms, 2]  dtype=torch.long
    edge_index : [2, N_edges]  dtype=torch.long
    edge_attr  : [N_edges, 2]  dtype=torch.long
    batch      : [N_atoms]     dtype=torch.long

Architecture:
    5-layer GINConv (add aggregation, MLP update) + batch norm
    Mean pooling -> [B, 300] -> Linear -> [B, 1] logit
    Requires pretrained weights (supervised_contextpred.pth)

Training:
    MAML (same as GCN)

Evaluation:
    ROC-AUC
```

### Model: GraphSAGE (FS-GNNTR baseline mode)

```
Model: GraphSAGE

Location:
    gnntr_train.py   (GNNTR class with baseline=1, gnn_type="graphsage")
    gnn_models.py    (GraphSAGEConv, GNN)
    transformer.py   (GNN_prediction with graphsage backbone)

Input:
    graph only

Uses:
    x          : [N_atoms, 2]  dtype=torch.long
    edge_index : [2, N_edges]  dtype=torch.long
    edge_attr  : [N_edges, 2]  dtype=torch.long
    batch      : [N_atoms]     dtype=torch.long

Architecture:
    5-layer GraphSAGEConv (mean aggregation, L2 norm) + batch norm
    Mean pooling -> [B, 300] -> Linear -> [B, 1] logit
    Requires pretrained weights (graphsage_supervised_contextpred.pth)

Training:
    MAML (same protocol)

Evaluation:
    ROC-AUC
```

### Model: GAT (selectable, not in original eval scripts)

```
Model: GAT

Location:
    gnn_models.py    (GATConv, GNN with gnn_type="gat")
    transformer.py   (GNN_prediction with gat backbone)

Input:
    graph only

Uses:
    x          : [N_atoms, 2]  dtype=torch.long
    edge_index : [2, N_edges]  dtype=torch.long
    edge_attr  : [N_edges, 2]  dtype=torch.long

Architecture:
    5-layer GATConv (2-head attention) + batch norm
    Mean pooling -> [B, 300] -> Linear -> [B, 1]
    Requires pretrained weights (gat_supervised_contextpred.pth)

Training:
    MAML (same protocol)

Evaluation:
    ROC-AUC
```

> **Note on "FS-GCvTR", "Meta-MGNN", "MAML", "Seq2Seq", "EGNN"**:
> These are NOT directly implemented in the FS-GNNTR repository. The `baseline.txt` file
> in the 3Br-MGD project contains pasted code for "FS-GNNConv" variants, but no separate
> EGNN, Meta-MGNN, or Seq2Seq implementations exist in the cloned repository.
> Decision required: either source these from separate repositories or exclude them from scope.

---

## 0.3 Architecture Dependency Analysis

| Model | Graph | Fingerprint | Sequence | Transformer | Pre-training |
|-------|-------|-------------|----------|-------------|--------------|
| FS-GNNTR | Yes | No | No | Yes (Vision TR, patch=30x1) | Yes |
| GCN | Yes | No | No | No | Yes |
| GIN | Yes | No | No | No | Yes |
| GraphSAGE | Yes | No | No | No | Yes |
| GAT | Yes | No | No | No | Yes |
| 3Br-MGD | Yes | Yes | Yes | Yes (Multi-head Attn) | No |

---

## 0.4 Dataset Dependency Analysis

### Original FS-GNNTR Dataset Loading

**Preprocessing pipeline:**
1. `split_into_directories(data)` — reads raw CSV, **unseeded** `np.random.shuffle`, splits per-task into binary pickles
2. `MoleculeDataset.process()` — loads pickle, calls `mol_to_graph_data_obj_simple()` per SMILES
3. `dataset()` — builds all task datasets

**Task partitioning (original):**
```
Tox21: 12 tasks total
  train: task_1 to task_9   (numeric index)
  test:  task_10, task_11, task_12

SIDER: 27 tasks total
  train: task_1 to task_21
  test:  task_22 to task_27
```

**Support/query generation (original):**
```python
random_sampler(D, d, t, k, n, train):
    # Uses hardcoded class size tables (datasets() function)
    # NO seed control
    # s_pos = k random from positives (first data[t][0] items)
    # s_neg = k random from negatives (remaining items)
```

**Components that MUST be replaced:**
| Component | Reason |
|-----------|--------|
| Task partitioning | Must use 3Br-MGD named task lists |
| `random_sampler()` | No seed; uses hardcoded tables; incompatible data format |
| `MoleculeDataset` | Incompatible with 3Br-MGD .pt format |
| `split_into_directories()` | Unseeded shuffle; wrong task boundaries |
| Graph feature encoding | long indices vs float one-hot |

---

## 0.5 Compatibility Analysis with 3Br-MGD

### `load_all_splits()` — Actual Signature

```python
# File: 3Br_MGD/Br_MGD/data.py (Lines 305-329)
def load_all_splits(data_dir: str) -> tuple:
    """
    Args:
        data_dir: str — directory with 'dataset_info.json',
                        'meta_train/' and 'meta_test/' subdirs

    Returns:
        meta_train: dict[str, dict]  {task_name: {'pos': [sample,...], 'neg': [sample,...]}}
        meta_test:  dict[str, dict]  {task_name: {'pos': [sample,...], 'neg': [sample,...]}}
    """
```

### Sample Schema — Actual Fields

```python
sample = {
    'fp':       torch.Tensor,          # shape=[2048],  dtype=float32
                                       # Morgan fingerprint (radius=2, nBits=2048)
    'graph':    torch_geometric.data.Data,  # see graph schema below
    'sequence': torch.Tensor,          # shape=[200],   dtype=torch.long
                                       # SMILES tokenized via SMILESVocabulary
    'label':    int,                   # 0 or 1
    'smiles':   str,                   # raw SMILES string
}
```

Tasks stored as: `{'pos': [sample, ...], 'neg': [sample, ...]}`

### Graph Schema — Actual

```python
graph.x          # shape=[N_atoms, 78],   dtype=torch.float32
                 # one-hot: 44 atom symbols + 11 degree + 11 H + 11 valence + 1 aromatic

graph.edge_index # shape=[2, N_edges],    dtype=torch.long
                 # COO format, bidirectional edges

graph.edge_attr  # shape=[N_edges, 8],    dtype=torch.float32
                 # one-hot: 4 bond types + 4 stereo types
```

---

## 0.6 Feature Mismatch Analysis

### Ground Truth: 3Br-MGD Graph

```
node features: [N, 78]  float32  (one-hot encoded, 78 dimensions)
edge features: [N, 8]   float32  (one-hot encoded, 8 dimensions)
```

### FS-GNNTR / All GNN Baselines Expected

```
node features: [N, 2]   long     (integer index pair: atom_num_idx, chirality_idx)
edge features: [N, 2]   long     (integer index pair: bond_type_idx, bond_dir_idx)

GNN forward: x = x_embedding1(x[:,0]) + x_embedding2(x[:,1])
             -> requires LONG integer indices, NOT float one-hot
```

### Mismatch Summary

| Property | 3Br-MGD | FS-GNNTR Baselines | Verdict |
|----------|---------|-------------------|---------|
| x dtype | float32 | long (int) | **INCOMPATIBLE** |
| x shape | [N, 78] | [N, 2] | **INCOMPATIBLE** |
| edge_attr dtype | float32 | long (int) | **INCOMPATIBLE** |
| edge_attr shape | [N, 8] | [N, 2] | **INCOMPATIBLE** |

### Proposed Adaptation Strategy (All FS-GNNTR Baselines)

Constraint: Do NOT modify model architectures. Adapt data to architecture.

**Strategy: Graph Format Adapter**

Write an `adapter.py` per baseline that converts 3Br-MGD float graphs to FS-GNNTR long-index format:

```python
# adapter.py  (pseudocode — implementation pending approval)

# Atom type mapping: 3Br-MGD uses 44-element atom symbol list
# FS-GNNTR uses allowable_features['possible_atomic_num_list'] (atomic number 1-118)
# Need a mapping from 3Br-MGD symbol position -> atomic number -> FS-GNNTR index

ATOM_SYMBOLS_3BR = [
    'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As',
    'Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti',
    'Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt',
    'Hg','Pb','Unknown'
]

def adapt_graph_to_fsgnntr(graph_3br):
    """
    Convert 3Br-MGD graph (float one-hot) to FS-GNNTR graph (long index).
    Does NOT modify any model. Only transforms data format.
    """
    # Recover atom symbol index from one-hot
    atom_onehot = graph_3br.x[:, :44]           # [N, 44] float
    atom_idx_3br = atom_onehot.argmax(dim=1)     # [N] int, 0-43

    # Map to FS-GNNTR's possible_atomic_num_list index
    # (requires symbol -> atomic_num -> fsgnntr_index lookup)
    atom_num_idx = map_to_fsgnntr_atom_idx(atom_idx_3br)  # [N] long
    chirality_idx = torch.zeros(atom_num_idx.shape[0], dtype=torch.long)  # default: UNSPECIFIED

    x_new = torch.stack([atom_num_idx, chirality_idx], dim=1)  # [N, 2] long

    # Recover bond type from one-hot
    bond_onehot = graph_3br.edge_attr[:, :4]    # [E, 4] float
    bond_type_idx = bond_onehot.argmax(dim=1)   # [E] long (0=SINGLE,1=DOUBLE,2=TRIPLE,3=AROMATIC)
    bond_dir_idx = torch.zeros(bond_type_idx.shape[0], dtype=torch.long)  # default: NONE

    edge_attr_new = torch.stack([bond_type_idx, bond_dir_idx], dim=1)  # [E, 2] long

    return Data(
        x=x_new,
        edge_index=graph_3br.edge_index,
        edge_attr=edge_attr_new,
    )
```

### Additional Critical Issues

1. **Hardcoded batch size in TR**: `transformer.py:105` has `emb.reshape(10, 1, 300, 1)`. The Transformer forward pass requires exactly batch_size=10. This constrains how support/query batches must be padded/truncated.

2. **Pre-trained weights**: All GNN baselines call `gnn.from_pretrained(pretrained_path)`. These weights are mandatory for fair comparison with original paper results. They are included in `pre-trained/`.

3. **Loss function**: FS-GNNTR uses `BCEWithLogitsLoss` (binary, sigmoid output). 3Br-MGD uses prototypical networks with `CrossEntropyLoss` (softmax). Each baseline must keep its own loss function.

4. **Task identity mapping**: FS-GNNTR task_10, task_11, task_12 correspond to Tox21 SR-HSE, SR-MMP, SR-p53 (same as 3Br-MGD meta_test). Must verify exact SMILES-level correspondence between the two preprocessing pipelines.

---

*End of Phase 0 Analysis — Awaiting user approval before proceeding to implementation.*
