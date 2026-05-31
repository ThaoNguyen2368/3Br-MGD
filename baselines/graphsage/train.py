"""
train.py — Meta-training entry point for GraphSAGE baseline.
Delegates to shared gnn_baseline_runner.run_train().

Usage:
  conda activate 3Br_MGD
  python baselines/graphsage/train.py \
      --data_dir 3Br_MGD/Data/tox21/processed \
      --dataset tox21 \
      --shots 5 10
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import config as CFG
from gnn_baseline_runner import run_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Meta-train {CFG.MODEL_NAME} baseline')
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default=CFG.CHECKPOINT_DIR)
    parser.add_argument('--dataset',        type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',          type=int, nargs='+', default=[5, 10])
    parser.add_argument('--max_epochs',     type=int, default=CFG.MAX_EPOCHS)
    parser.add_argument('--patience',       type=int, default=CFG.PATIENCE)
    parser.add_argument('--train_episodes', type=int, default=CFG.TRAIN_EPISODES)
    parser.add_argument('--q_query',        type=int, default=CFG.Q_QUERY)
    parser.add_argument('--seed',           type=int, default=CFG.SEED)
    args = parser.parse_args()
    run_train(CFG, args)
