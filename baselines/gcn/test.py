"""
test.py — Meta-testing entry point for GCN baseline.
Delegates to shared gnn_baseline_runner.run_test().

Usage:
  conda activate 3Br_MGD
  python baselines/gcn/test.py \
      --data_dir 3Br_MGD/Data/tox21/processed \
      --dataset tox21 \
      --shots 5 10 \
      --episodes_file baselines/episodes_seed42_tox21.json
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import config as CFG
from gnn_baseline_runner import run_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Meta-test {CFG.MODEL_NAME} baseline')
    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--output_dir',    type=str, default=CFG.CHECKPOINT_DIR)
    parser.add_argument('--dataset',       type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',         type=int, nargs='+', default=[5, 10])
    parser.add_argument('--checkpoint',    type=str, default=None)
    parser.add_argument('--episodes_file', type=str, required=True)
    parser.add_argument('--seed',          type=int, default=CFG.SEED)
    args = parser.parse_args()
    run_test(CFG, args)
