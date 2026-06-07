"""
config.py — Hyperparameters for FS-GNNTR baseline.

FS-GNNTR = GNN_prediction (5-layer GIN) + TR (Vision Transformer) + MAML.
Architecture is unchanged from original paper.
Only data loading, task split, episodes, and stopping criterion are replaced.
"""

import os

# ── Repository path ─────────────────────────────────────────────────────────────
# Pre-trained weights are in baselines/pre-trained/ (self-contained, no external repo needed)
PRETRAINED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'pre-trained')
)

# ── Architecture (DO NOT CHANGE) ────────────────────────────────────────────────
GNN_TYPE      = 'gin'          # backbone: 'gin', 'gcn', 'graphsage', 'gat'
EMB_SIZE      = 300            # GNN embedding dimension
GRAPH_LAYERS  = 5              # number of GNN layers
JK            = 'last'         # Jumping Knowledge aggregation
DROPOUT       = 0.5            # dropout probability
POOLING       = 'mean'         # graph pooling

# Transformer config (TR class)
TR_EMB_SIZE   = 300            # must match GNN EMB_SIZE
TR_PATCH_SIZE = (30, 1)        # patch size for vision transformer
TR_NUM_CLS    = 1              # number of output classes
TR_DIM        = 128
TR_DEPTH      = 5
TR_HEADS      = 5
TR_MLP_DIM    = 256

# ── Pre-trained weights ─────────────────────────────────────────────────────────
# Stored in baselines/pre-trained/ (included in repository)
PRETRAINED_GIN  = os.path.join(PRETRAINED_DIR, 'supervised_contextpred.pth')
PRETRAINED_GCN  = os.path.join(PRETRAINED_DIR, 'gcn_supervised_contextpred.pth')
PRETRAINED_SAGE = os.path.join(PRETRAINED_DIR, 'graphsage_supervised_contextpred.pth')
PRETRAINED_GAT  = os.path.join(PRETRAINED_DIR, 'gat_supervised_contextpred.pth')

PRETRAINED = PRETRAINED_GIN   # default for FS-GNNTR (GIN backbone)

# ── MAML hyperparameters (original paper values) ────────────────────────────────
LR_GNN        = 0.001          # meta-learning rate for GNN optimizer
LR_TR         = 1e-5           # meta-learning rate for Transformer optimizer
LR_UPDATE     = 0.5            # MAML inner update step size
N_INNER_TRAIN = 1              # MAML inner steps during training
N_INNER_TEST  = 20             # MAML adaptation steps during testing (k_test)

# ── Meta-learning settings ───────────────────────────────────────────────────────
K_SHOT_DEFAULT = 10            # default shots (can be overridden via CLI)
Q_QUERY        = 256           # query samples per class during training

# ── Training protocol (matches 3Br-MGD for fairness) ───────────────────────────
MAX_EPOCHS      = 200
PATIENCE        = 20
TRAIN_EPISODES  = 100          # episodes per epoch

# ── Loss ─────────────────────────────────────────────────────────────────────────
# pos_weight: 25 for tox21 (class imbalance), 1 for sider
POS_WEIGHT_TOX21  = 25.0
POS_WEIGHT_SIDER  = 1.0

# ── Output ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'fsgnntr')
)

# ── Seed ────────────────────────────────────────────────────────────────────────
SEED = 42
