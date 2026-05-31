"""
config.py — Hyperparameters for GCN baseline.

GCN = GNN_prediction (5-layer GCN, emb=300) + MAML (no Transformer).
Architecture is unchanged from FS-GNNTR paper's GCN baseline (baseline=1, gnn_type="gcn").
"""

import os

MODEL_NAME = 'GCN'

FSGNNTR_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'FS-GNNTR_repo', 'FS-GNNTR')
)

# ── Architecture (DO NOT CHANGE) ────────────────────────────────────────────────
GNN_TYPE     = 'gcn'
EMB_SIZE     = 300
GRAPH_LAYERS = 5
JK           = 'last'
DROPOUT      = 0.5
POOLING      = 'mean'

# ── Pre-trained weights ─────────────────────────────────────────────────────────
PRETRAINED = os.path.join(FSGNNTR_REPO, 'pre-trained', 'gcn_supervised_contextpred.pth')

# ── MAML hyperparameters (original paper values) ────────────────────────────────
LR_GNN        = 0.001
LR_UPDATE     = 0.5
N_INNER_TRAIN = 1
N_INNER_TEST  = 20

# ── Meta-learning settings ───────────────────────────────────────────────────────
Q_QUERY = 256

# ── Training protocol (matches 3Br-MGD) ────────────────────────────────────────
MAX_EPOCHS     = 200
PATIENCE       = 20
TRAIN_EPISODES = 100

# ── Output ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'gcn')
)

SEED = 42
