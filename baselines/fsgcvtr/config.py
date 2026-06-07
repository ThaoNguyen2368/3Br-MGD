"""
config.py — Hyperparameters for FS-GCvTR baseline.

FS-GCvTR = GNN_prediction (5-layer GIN) + ConvTR (Convolutional Transformer) + MAML.
Architecture is unchanged from original paper.
Only data loading, task split, episodes, and stopping criterion are replaced.
"""

import os

# ── Repository path ─────────────────────────────────────────────────────────────
PRETRAINED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'pre-trained')
)

# ── Architecture (DO NOT CHANGE) ────────────────────────────────────────────────
GNN_TYPE      = 'gin'          # backbone: 'gin', 'gcn', 'graphsage'
EMB_SIZE      = 300            # GNN embedding dimension
GRAPH_LAYERS  = 5              # number of GNN layers
JK            = 'last'         # Jumping Knowledge aggregation
DROPOUT       = 0.5            # dropout probability
POOLING       = 'mean'         # graph pooling

# ── Pre-trained weights ─────────────────────────────────────────────────────────
# Stored in baselines/pre-trained/ (included in repository)
PRETRAINED_GIN  = os.path.join(PRETRAINED_DIR, 'supervised_contextpred.pth')
PRETRAINED      = PRETRAINED_GIN   # default for FS-GCvTR (GIN backbone)

# ── MAML hyperparameters (original paper values) ────────────────────────────────
LR_GNN        = 0.001          # meta-learning rate for GNN optimizer
LR_TR         = 1e-5           # meta-learning rate for Transformer optimizer
LR_UPDATE     = 0.4            # MAML inner update step size
N_INNER_TRAIN = 5              # MAML inner steps during training (k_train)
N_INNER_TEST  = 10             # MAML adaptation steps during testing (k_test)

# ── Meta-learning settings ───────────────────────────────────────────────────────
K_SHOT_DEFAULT = 10            # default shots (can be overridden via CLI)
Q_QUERY        = 128           # query samples per class during training

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
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'fsgcvtr')
)

# ── Seed ────────────────────────────────────────────────────────────────────────
SEED = 42
