import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from data import load_all_splits
from branch_Build import build_model, VARIANT_NAMES, VARIANT_INFO
from BrMGD_eval import evaluate_meta_task
from BrMGD_train import create_meta_task

def evaluate_branch(
    protonet,
    meta_test: dict,
    device: torch.device,
    K_shot: int,
    Q_query: int,
    test_episodes: int,
    variant: str,
    dataset: str,
    shot_name: str,
    result_dir: str,
) -> dict:
    
    os.makedirs(result_dir, exist_ok=True)
    WRITE_LIMIT = 30

    task_names = list(meta_test.keys())
    n_tasks    = len(task_names)

 
    exp     = [[] for _ in range(n_tasks)]   # AUROC
    acc_exp = [[] for _ in range(n_tasks)]
    f1_exp  = [[] for _ in range(n_tasks)]

    shot_num     = shot_name.replace('-shot', '')
    episode_file = os.path.join(
        result_dir,
        f"mean-{variant}-{dataset}-{shot_num}shot.txt"
    )

    with open(episode_file, 'a', encoding='utf-8') as rf:
        rf.write(f"\n{'='*65}\n")
        rf.write(f"Variant : {VARIANT_INFO[variant]['name']}  |  {shot_name}\n")
        rf.write(f"Dataset : {dataset}  |  K_shot={K_shot}\n")
        rf.write(f"Tasks   : {task_names}\n")
        rf.write(f"{'='*65}\n")

    print(f"  Running {test_episodes} test episodes...")

    for ep in range(1, test_episodes + 1):

        for i, task_name in enumerate(task_names):
            task_data = meta_test[task_name]
            try:
                support, query = create_meta_task(
                    task_data, K_shot, Q_query, train=False
                )
                acc, f1, auroc = evaluate_meta_task(
                    protonet, support, query, device
                )
                exp[i].append(round(auroc, 4) if not np.isnan(auroc) else 0.0)
                if not np.isnan(acc): acc_exp[i].append(acc)
                if not np.isnan(f1):  f1_exp[i].append(f1)
            except Exception:
                exp[i].append(0.0)

        if ep <= WRITE_LIMIT:
            with open(episode_file, 'a', encoding='utf-8') as rf:
                rf.write(f"Results: \t{exp}\n")

        # Log 
        if ep % 10 == 0:
            current_means = [
                f"{np.mean(exp[i]):.4f}" if exp[i] else 'nan'
                for i in range(n_tasks)
            ]
            print(f"    Episode {ep:3d}/{test_episodes} — AUROC: {current_means}")

    results = {}
    summary_lines = []
    summary_lines.append(f"\n{'-'*65}")
    summary_lines.append(
        f"{'Task':<40} {'Acc':>10} {'F1':>10} {'AUROC':>10}  (mean±std)"
    )
    summary_lines.append(f"{'-'*65}")

    for i, task_name in enumerate(task_names):
        auroc_mean = np.mean(exp[i])      if exp[i]     else float('nan')
        auroc_std  = np.std(exp[i])       if len(exp[i])     > 1 else 0.0
        acc_mean   = np.mean(acc_exp[i])  if acc_exp[i] else float('nan')
        acc_std    = np.std(acc_exp[i])   if len(acc_exp[i]) > 1 else 0.0
        f1_mean    = np.mean(f1_exp[i])   if f1_exp[i]  else float('nan')
        f1_std     = np.std(f1_exp[i])    if len(f1_exp[i])  > 1 else 0.0

        line = (f"{task_name:<40} "
                f"{acc_mean:.4f}±{acc_std:.4f}  "
                f"{f1_mean:.4f}±{f1_std:.4f}  "
                f"{auroc_mean:.4f}±{auroc_std:.4f}")
        summary_lines.append(line)
        print(f"    {line}")

        results[task_name] = {
            'auroc':     auroc_mean,
            'auroc_std': auroc_std,
            'acc':       acc_mean,
            'acc_std':   acc_std,
            'f1':        f1_mean,
            'f1_std':    f1_std,
            'raw_auroc': exp[i],
        }

    all_auroc = [results[t]['auroc'] for t in task_names if not np.isnan(results[t]['auroc'])]
    all_acc   = [results[t]['acc']   for t in task_names if not np.isnan(results[t]['acc'])]
    all_f1    = [results[t]['f1']    for t in task_names if not np.isnan(results[t]['f1'])]

    ov_acc   = np.mean(all_acc)   if all_acc   else float('nan')
    ov_f1    = np.mean(all_f1)    if all_f1    else float('nan')
    ov_auroc = np.mean(all_auroc) if all_auroc else float('nan')

    ov_line = (f"\n{'OVERALL AVERAGE':<40} "
               f"{ov_acc:.4f}{'':>14}"
               f"{ov_f1:.4f}{'':>14}"
               f"{ov_auroc:.4f}")
    summary_lines.append(ov_line)
    summary_lines.append('-'*65)
    print(ov_line)

    with open(episode_file, 'a', encoding='utf-8') as rf:
        rf.write('\n'.join(summary_lines) + '\n')

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate branch variants on meta-test tasks'
    )
    parser.add_argument('--data_dir',        type=str, required=True,
                        help='Directory processed data')
    parser.add_argument('--checkpoint_dir',  type=str, default='checkpoints',
                        help='Directory include checkpoint .pth')
    parser.add_argument('--dataset',         type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--variants',        type=str, nargs='+', default=['all'],
                        help=f"List variant or 'all'. "
                             f"choose from: {VARIANT_NAMES}")
    parser.add_argument('--shots',           type=int, nargs='+', default=[5, 10])
    parser.add_argument('--test_episodes',   type=int, default=30,
                        help='episode test (default 100)')
    parser.add_argument('--q_query',         type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.variants == ['all']:
        variants_to_run = VARIANT_NAMES
    else:
        variants_to_run = []
        for v in args.variants:
            if v not in VARIANT_NAMES:
                print(f"WARNING: variant '{v}' skip.")
            else:
                variants_to_run.append(v)

    if not variants_to_run:
        print("Invalid.")
        return

    print(f"Device   : {device}")
    print(f"Dataset  : {args.dataset}")
    print(f"Shots    : {args.shots}")
    print(f"Variants : {variants_to_run}")

    os.makedirs('results', exist_ok=True)

    _, _, meta_test = load_all_splits(args.data_dir)

    # all_results[shot_name][variant] = {task: metrics}
    all_results = {}

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        all_results[shot_name] = {}

        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")

        for variant in variants_to_run:
            info = VARIANT_INFO[variant]

            ckpt_name = f"{variant}_{args.dataset}_{shot_name}_best.pth"
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

            if not os.path.exists(ckpt_path):
                print(f"  WARNING: Checkpoint do not exist: {ckpt_path}. Skipping.")
                continue

            print(f"\n  ── {info['name']} ──")
            print(f"     Loading: {ckpt_path}")

            # Rebuild model and load weights
            ckpt     = torch.load(ckpt_path, map_location=device)
            protonet = build_model(variant, device)
            protonet.load_state_dict(ckpt['model_state'])
            print(f"     epoch={ckpt.get('epoch','?')}, "
                  f"val_auroc={ckpt.get('val_auroc', 0):.4f}")

            # Evaluate
            task_results = evaluate_branch(
                protonet      = protonet,
                meta_test     = meta_test,
                device        = device,
                K_shot        = K_shot,
                Q_query       = args.q_query,
                test_episodes = args.test_episodes,
                variant       = variant,
                dataset       = args.dataset,
                shot_name     = shot_name,
                result_dir    = 'results',
            )

            all_results[shot_name][variant] = task_results

            aurocs = [r['auroc'] for r in task_results.values()
                      if not np.isnan(r['auroc'])]
            if aurocs:
                print(f"     Mean AUROC = {np.mean(aurocs):.4f}")

    json_path = os.path.join('results', f'ablation_{args.dataset}_summary.json')
    json_data = {}
    for shot_name, variant_results in all_results.items():
        json_data[shot_name] = {}
        for variant, task_results in variant_results.items():
            aurocs = [r['auroc'] for r in task_results.values()
                      if not np.isnan(r['auroc'])]
            json_data[shot_name][variant] = {
                'variant_name': VARIANT_INFO[variant]['name'],
                'mean_auroc':   float(np.mean(aurocs)) if aurocs else None,
                'per_task': {
                    t: {k: float(v) if not isinstance(v, list) else v
                        for k, v in r.items()
                        if k != 'raw_auroc'}
                    for t, r in task_results.items()
                },
            }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\nSummary JSON → {json_path}")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()