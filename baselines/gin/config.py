"""
config.py — Hyperparameters for GIN baseline.

GIN = GNN_prediction (5-layer GIN, emb=300) + MAML (no Transformer).
Architecture is unchanged from FS-GNNTR paper's GIN baseline (baseline=1, gnn_type="gin").
"""

import os

MODEL_NAME = 'GIN'

PRETRAINED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'pre-trained')
)

# ── Architecture (DO NOT CHANGE) ────────────────────────────────────────────────
GNN_TYPE     = 'gin'
EMB_SIZE     = 300
GRAPH_LAYERS = 5
JK           = 'last'
DROPOUT      = 0.5
POOLING      = 'mean'

# ── Pre-trained weights ─────────────────────────────────────────────────────────
PRETRAINED = os.path.join(PRETRAINED_DIR, 'supervised_contextpred.pth')

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
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'gin')
)

SEED = 42
