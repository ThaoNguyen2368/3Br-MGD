"""
config.py — Hyperparameters for GraphSAGE baseline.

GraphSAGE = GNN_prediction (5-layer GraphSAGE, emb=300) + MAML (no Transformer).
Architecture is unchanged from FS-GNNTR paper's GraphSAGE baseline (baseline=1, gnn_type="graphsage").
"""

import os

MODEL_NAME = 'GraphSAGE'

PRETRAINED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'pre-trained')
)

# ── Architecture (DO NOT CHANGE) ────────────────────────────────────────────────
GNN_TYPE     = 'graphsage'
EMB_SIZE     = 300
GRAPH_LAYERS = 5
JK           = 'last'
DROPOUT      = 0.5
POOLING      = 'mean'

# ── Pre-trained weights ─────────────────────────────────────────────────────────
PRETRAINED = os.path.join(PRETRAINED_DIR, 'graphsage_supervised_contextpred.pth')

# ── MAML hyperparameters ─────────────────────────────────────────────────────────
LR_GNN        = 0.001
LR_UPDATE     = 0.5
N_INNER_TRAIN = 1
N_INNER_TEST  = 20

# ── Meta-learning settings ───────────────────────────────────────────────────────
Q_QUERY = 256

# ── Training protocol ───────────────────────────────────────────────────────────
MAX_EPOCHS     = 200
PATIENCE       = 20
TRAIN_EPISODES = 100

# ── Output ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'graphsage')
)

SEED = 42
