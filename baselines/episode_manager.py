"""
episode_manager.py — Shared episode pre-generation and loading.

All baselines use the SAME pre-generated test episodes.
No model may generate its own support/query splits for meta_test.

Usage:
  # Generate once (run from project root D:\\3Br_MGD):
  python baselines/episode_manager.py \
      --generate \
      --data_dir 3Br_MGD/Data/tox21/processed \
      --dataset tox21 \
      --shots 5 10 \
      --n_episodes 30 \
      --seed 42

  # Load in test scripts:
  from baselines.episode_manager import load_episodes
  episodes = load_episodes("baselines/episodes_seed42_tox21.json")
"""

import os
import sys
import json
import argparse

# ─── Path setup so we can import from 3Br_MGD/Br_MGD ──────────────────────────
# IMPORTANT: Insert at index 0 to override any FS-GNNTR data.py that may already
# be in sys.path. Both repos have a 'data.py'; we need 3Br-MGD's version here.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_BRMGD_PATH   = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')
# Always insert at 0 to guarantee priority over FS-GNNTR's data.py
if _BRMGD_PATH in sys.path:
    sys.path.remove(_BRMGD_PATH)
sys.path.insert(0, _BRMGD_PATH)

from seed_utils import set_seed
from data import load_all_splits
from BrMGD_train import create_meta_task


def generate_episodes(
    data_dir: str,
    dataset: str,
    K_shots: list,
    n_episodes: int,
    seed: int = 42,
) -> dict:
    """
    Pre-generate shared test episodes from 3Br-MGD meta_test tasks.

    Args:
        data_dir   : Path to preprocessed data directory (contains dataset_info.json)
        dataset    : 'tox21' or 'sider'
        K_shots    : list of K values, e.g. [5, 10]
        n_episodes : Number of episodes per task per K_shot
        seed       : Random seed (fixed at 42)

    Returns:
        episodes dict (also saved to disk)
    """
    set_seed(seed)

    _, meta_test = load_all_splits(data_dir)
    task_names = list(meta_test.keys())
    print(f"Meta-test tasks ({len(task_names)}): {task_names}")

    episodes = {
        'dataset':     dataset,
        'seed':        seed,
        'n_episodes':  n_episodes,
        'K_shots':     K_shots,
        'meta_test_tasks': task_names,
    }

    for K_shot in K_shots:
        key = f"{K_shot}-shot"
        episodes[key] = []

        for ep_idx in range(n_episodes):
            for task_name in task_names:
                task_data = meta_test[task_name]
                try:
                    support, query = create_meta_task(
                        task_data, K_shot, Q_query=None, train=False
                    )
                    episodes[key].append({
                        'task':          task_name,
                        'episode_idx':   ep_idx,
                        'support_smiles': [s['smiles'] for s in support],
                        'support_labels': [s['label']  for s in support],
                        'query_smiles':  [q['smiles'] for q in query],
                        'query_labels':  [q['label']  for q in query],
                    })
                except Exception as e:
                    print(f"  WARNING: episode {ep_idx}, task {task_name}: {e}")

        print(f"  {key}: generated {len(episodes[key])} (task, episode) pairs")

    return episodes


def save_episodes(episodes: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    print(f"Episodes saved → {out_path}")


def load_episodes(path: str) -> dict:
    """Load pre-generated episodes from JSON file."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


_PRECOMPUTED_GRAPHS = {}

def build_smiles_lookup(data_dir: str):
    """
    Load all .pt files from data_dir and map SMILES to their precomputed PyG graphs.
    Call this once before testing to bypass RDKit entirely.
    """
    from data import load_all_splits
    print(f"Preloading graphs from {data_dir} to bypass RDKit...")
    meta_train, meta_test = load_all_splits(data_dir)
    
    for splits in [meta_train, meta_test]:
        for task_name, task_data in splits.items():
            for sample in task_data['pos'] + task_data['neg']:
                # The .pt file stores 'smiles' and 'graph'
                if sample['smiles'] not in _PRECOMPUTED_GRAPHS:
                    _PRECOMPUTED_GRAPHS[sample['smiles']] = sample['graph']
    print(f"Preloaded {len(_PRECOMPUTED_GRAPHS)} unique graphs.")


def reconstruct_sample_from_smiles(smiles: str, label: int) -> dict:
    """
    Rebuild a minimal 3Br-MGD sample from a SMILES string.
    Returns {'graph': Data(float, 3Br-MGD format), 'label': int, 'smiles': str}
    """
    if smiles in _PRECOMPUTED_GRAPHS:
        graph = _PRECOMPUTED_GRAPHS[smiles]
    else:
        # Fallback if graph wasn't preloaded (e.g. not called build_smiles_lookup)
        from data import smiles_to_graph
        graph = smiles_to_graph(smiles)
        _PRECOMPUTED_GRAPHS[smiles] = graph
        
    if graph is None:
        return None
    return {
        'graph':  graph,
        'label':  label,
        'smiles': smiles,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate shared test episodes for all baselines')
    parser.add_argument('--generate',   action='store_true', help='Generate episodes')
    parser.add_argument('--data_dir',   type=str, required=True,
                        help='Path to preprocessed data dir (contains dataset_info.json)')
    parser.add_argument('--dataset',    type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',      type=int, nargs='+', default=[5, 10])
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--out_dir',    type=str, default='baselines',
                        help='Directory to save episode JSON files')
    args = parser.parse_args()

    if args.generate:
        episodes = generate_episodes(
            data_dir   = args.data_dir,
            dataset    = args.dataset,
            K_shots    = args.shots,
            n_episodes = args.n_episodes,
            seed       = args.seed,
        )
        out_path = os.path.join(
            args.out_dir,
            f"episodes_seed{args.seed}_{args.dataset}.json"
        )
        save_episodes(episodes, out_path)
    else:
        parser.print_help()
