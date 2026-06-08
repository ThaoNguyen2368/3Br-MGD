import os

MODEL_NAME = 'AttFPGNN'

ATTFPGNN_REPO = os.path.dirname(__file__)
ADKF_IFT_DIR = os.path.join(ATTFPGNN_REPO, 'chem_lib')
ATTFPGNN_DATA_DIR = os.path.join(ATTFPGNN_REPO, 'data')

# Architecture (defaults from AttFPGNN-MAML)
GNN_TYPE = 'gin'
GRAPH_LAYERS = 5
EMB_DIM = 300
DROPOUT = 0.5
JK = 'last'
POOLING = 'mean'
MAP_DIM = 128
PRETRAINED = False

# MAML
META_LR = 0.0005
INNER_LR = 0.01
INNER_UPDATE_STEP = 5
UPDATE_STEP_TEST = 10
WEIGHT_DECAY = 5e-5

# Training
MAX_EPOCHS = 1000
PATIENCE = 100
BATCH_TASK = 10
Q_QUERY = 32

CHECKPOINT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'attfpgnn')
)

SEED = 42
