"""
summarize_results.py — Aggregate and compare all baseline results.

Usage (from D:\\3Br_MGD):
  conda run -n 3Br_MGD python baselines/summarize_results.py --dataset tox21
  conda run -n 3Br_MGD python baselines/summarize_results.py --dataset sider
  conda run -n 3Br_MGD python baselines/summarize_results.py --dataset all
"""

import os
import sys
import json
import argparse
import glob

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_result(path):
    with open(path) as f:
        return json.load(f)


def format_table(dataset, all_results):
    """Print a comparison table: rows=models, cols=shot x task."""
    print(f"\n{'='*70}")
    print(f"  RESULTS: {dataset.upper()}")
    print(f"{'='*70}")

    # Collect all shot configs
    shot_keys = set()
    for r in all_results.values():
        shot_keys.update(r.get('shots', {}).keys())
    shot_keys = sorted(shot_keys)

    # Header
    print(f"  {'Model':<15}", end='')
    for sk in shot_keys:
        print(f"  {sk:>20}", end='')
    print()
    print(f"  {'-'*15}", end='')
    for sk in shot_keys:
        print(f"  {'─'*20}", end='')
    print()

    # Rows
    for model_name, result in sorted(all_results.items()):
        print(f"  {model_name:<15}", end='')
        for sk in shot_keys:
            shot_data = result.get('shots', {}).get(sk, {})
            if shot_data:
                m = shot_data.get('auc_mean', float('nan'))
                s = shot_data.get('auc_std', 0.0)
                print(f"  {m:.4f} ± {s:.4f}    ", end='')
            else:
                print(f"  {'N/A':>20}", end='')
        print()

    print()

    # Per-task breakdown
    for sk in shot_keys:
        print(f"  --- {sk} per-task ---")
        # Collect all task names
        all_tasks = set()
        for r in all_results.values():
            per_task = r.get('shots', {}).get(sk, {}).get('per_task', {})
            all_tasks.update(per_task.keys())
        all_tasks = sorted(all_tasks)

        for task in all_tasks:
            print(f"    {task[:40]:<42}", end='')
            for model_name, result in sorted(all_results.items()):
                per_task = result.get('shots', {}).get(sk, {}).get('per_task', {})
                if task in per_task:
                    m = per_task[task]['auc_mean']
                    print(f"  {model_name[:8]}: {m:.4f}", end='')
                else:
                    print(f"  {model_name[:8]}: N/A   ", end='')
            print()
        print()


def process_dataset(dataset):
    # Look in results/<dataset>/ for results_<dataset>.json
    results_dir = os.path.join(_PROJECT_ROOT, 'results', dataset)

    # Also look in checkpoint directories as fallback
    search_dirs = [results_dir]
    for bl in ['fsgnntr', 'gcn', 'gin', 'graphsage', 'attfpgnn']:
        search_dirs.append(os.path.join(_PROJECT_ROOT, 'checkpoints', bl))

    all_results = {}

    for search_dir in search_dirs:
        pattern = os.path.join(search_dir, f'results_{dataset}.json')
        for path in glob.glob(pattern):
            r = load_result(path)
            model = r.get('model', os.path.basename(os.path.dirname(path)))
            all_results[model] = r
            print(f"  Loaded: {path}")

    if not all_results:
        print(f"  No results found for {dataset}. Run training + testing first.")
        return

    format_table(dataset, all_results)

    # Save combined summary
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Combined summary saved → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Summarize baseline results')
    parser.add_argument('--dataset', type=str, default='tox21',
                        choices=['tox21', 'sider', 'all'])
    args = parser.parse_args()

    datasets = ['tox21', 'sider'] if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        process_dataset(ds)


if __name__ == '__main__':
    main()
