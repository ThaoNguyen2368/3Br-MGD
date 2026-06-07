#!/usr/bin/env python
"""
run_all_baselines.py — Run training + testing for all baselines sequentially.

Usage (from D:\\3Br_MGD, conda env 3Br_MGD):
  conda run -n 3Br_MGD python baselines/run_all_baselines.py \
      --dataset tox21 --shots 5 10 --mode train
  conda run -n 3Br_MGD python baselines/run_all_baselines.py \
      --dataset tox21 --shots 5 10 --mode test

Or both at once:
  conda run -n 3Br_MGD python baselines/run_all_baselines.py \
      --dataset tox21 --shots 5 10 --mode all
"""

import os
import sys
import argparse
import subprocess
import copy

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

BASELINES = ['fsgcvtr', 'fsgnntr', 'gcn', 'gin', 'graphsage', 'attfpgnn']

TRAIN_SCRIPTS = {
    'fsgcvtr':   'baselines/fsgcvtr/train.py',
    'fsgnntr':   'baselines/fsgnntr/train.py',
    'gcn':       'baselines/gcn/train.py',
    'gin':       'baselines/gin/train.py',
    'graphsage': 'baselines/graphsage/train.py',
    'attfpgnn':  'baselines/attfpgnn/train.py',
}

TEST_SCRIPTS = {
    'fsgcvtr':   'baselines/fsgcvtr/test.py',
    'fsgnntr':   'baselines/fsgnntr/test.py',
    'gcn':       'baselines/gcn/test.py',
    'gin':       'baselines/gin/test.py',
    'graphsage': 'baselines/graphsage/test.py',
    'attfpgnn':  'baselines/attfpgnn/test.py',
}


def run_script(script_path, extra_args, label):
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    # Inject _PROJECT_ROOT into PYTHONPATH so child scripts can resolve
    # `import baselines.*` even before their own sys.path setup runs.
    env = copy.copy(os.environ)
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (_PROJECT_ROOT + os.pathsep + existing) if existing else _PROJECT_ROOT
    ret = subprocess.run(cmd, cwd=_PROJECT_ROOT, env=env)
    if ret.returncode != 0:
        print(f"  ERROR: {label} failed with exit code {ret.returncode}")
        return False
    print(f"  DONE: {label}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run all baselines')
    parser.add_argument('--dataset',   type=str, required=True, choices=['tox21', 'sider'])
    parser.add_argument('--shots',     type=int, nargs='+', default=[5, 10])
    parser.add_argument('--mode',      type=str, default='all',
                        choices=['train', 'test', 'all'])
    parser.add_argument('--baselines', type=str, nargs='+', default=BASELINES,
                        choices=BASELINES, help='Which baselines to run (default: all)')
    parser.add_argument('--max_epochs',type=int, default=200)
    parser.add_argument('--patience',  type=int, default=20)
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Data', args.dataset, 'processed')
    episodes_file = os.path.join(_PROJECT_ROOT, 'baselines',
                                 f'episodes_seed{args.seed}_{args.dataset}.json')

    shots_str = ' '.join(str(s) for s in args.shots)
    results_dir = os.path.join(_PROJECT_ROOT, 'results', args.dataset)

    failed = []

    for bl in args.baselines:
        if args.mode in ('train', 'all'):
            train_args = [
                '--data_dir',    data_dir,
                '--dataset',     args.dataset,
                '--shots',       *[str(s) for s in args.shots],
                '--max_epochs',  str(args.max_epochs),
                '--patience',    str(args.patience),
                '--seed',        str(args.seed),
            ]
            ok = run_script(TRAIN_SCRIPTS[bl], train_args,
                            f"TRAIN {bl.upper()} on {args.dataset} (shots={shots_str})")
            if not ok:
                failed.append(f"TRAIN:{bl}")

        if args.mode in ('test', 'all'):
            if not os.path.exists(episodes_file):
                print(f"  ERROR: Episodes file not found: {episodes_file}")
                print(f"  Run: python baselines/episode_manager.py --generate ...")
                failed.append(f"TEST:{bl} (no episodes)")
                continue

            test_args = [
                '--data_dir',       data_dir,
                '--output_dir',     results_dir,
                '--dataset',        args.dataset,
                '--shots',          *[str(s) for s in args.shots],
                '--episodes_file',  episodes_file,
                '--seed',           str(args.seed),
            ]
            ok = run_script(TEST_SCRIPTS[bl], test_args,
                            f"TEST {bl.upper()} on {args.dataset} (shots={shots_str})")
            if not ok:
                failed.append(f"TEST:{bl}")

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"ALL {'TRAINING' if args.mode=='train' else 'TESTING' if args.mode=='test' else 'TRAIN+TEST'} COMPLETE ✓")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
