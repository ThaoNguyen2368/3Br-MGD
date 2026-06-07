import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
# Thêm đường dẫn tới thư mục Br_MGD để import data và các hàm bổ trợ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Br_MGD")))

from data import load_all_splits, SMILES_VOCAB
from branch_Build import build_model, VARIANT_NAMES, VARIANT_INFO
from BrMGD_eval import evaluate_meta_task
from BrMGD_train import create_meta_task


def set_seed(seed: int = 42):
    """Fix tất cả nguồn ngẫu nhiên để đảm bảo test episodes reproducible."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_branch(
    protonet,
    ep_list: list,
    meta_test: dict,
    device: torch.device,
    K_shot: int,
    variant: str,
    dataset: str,
    shot_name: str,
    result_dir: str,
) -> dict:
    
    os.makedirs(result_dir, exist_ok=True)
    WRITE_LIMIT = 30

    # Pre-build SMILES lookup dictionary
    smiles_to_sample = {}
    for task_name, task_data in meta_test.items():
        smiles_to_sample[task_name] = {}
        for label in ['pos', 'neg']:
            for s in task_data[label]:
                if 'smiles' in s:
                    smiles_to_sample[task_name][s['smiles']] = s

    task_names = list(meta_test.keys())
    n_tasks    = len(task_names)
    
    task_to_idx = {t: i for i, t in enumerate(task_names)}
 
    exp       = [[] for _ in range(n_tasks)]   # AUROC
    auprc_exp = [[] for _ in range(n_tasks)]   # AUPRC
    acc_exp   = [[] for _ in range(n_tasks)]
    f1_exp    = [[] for _ in range(n_tasks)]

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
        rf.write(f"Total episodes={len(ep_list)}\n")
        rf.write(f"{'='*65}\n")

    print(f"  Running {len(ep_list)} test episodes...")

    for ep_idx, ep in enumerate(ep_list):
        task_name = ep['task']
        if task_name not in task_to_idx:
            continue
            
        i = task_to_idx[task_name]
        
        support = []
        for sm in ep['support_smiles']:
            if sm in smiles_to_sample[task_name]:
                support.append(smiles_to_sample[task_name][sm])
                
        query = []
        for sm in ep['query_smiles']:
            if sm in smiles_to_sample[task_name]:
                query.append(smiles_to_sample[task_name][sm])

        try:
            acc, f1, auroc, auprc = evaluate_meta_task(
                protonet, support, query, device
            )
            exp[i].append(round(auroc, 4) if not np.isnan(auroc) else 0.0)
            auprc_exp[i].append(round(auprc, 4) if not np.isnan(auprc) else 0.0)
            if not np.isnan(acc): acc_exp[i].append(acc)
            if not np.isnan(f1):  f1_exp[i].append(f1)
        except Exception:
            exp[i].append(0.0)
            auprc_exp[i].append(0.0)

        if ep_idx < WRITE_LIMIT * n_tasks:
            with open(episode_file, 'a', encoding='utf-8') as rf:
                rf.write(f"Task {task_name} AUROC: \t{exp[i][-1] if exp[i] else 0.0} \tAUPRC: \t{auprc_exp[i][-1] if auprc_exp[i] else 0.0}\n")

        # Log 
        if (ep_idx + 1) % (10 * n_tasks) == 0:
            current_means = [
                f"{np.mean(exp[idx]):.4f}" if exp[idx] else 'nan'
                for idx in range(n_tasks)
            ]
            print(f"    Processed {ep_idx + 1}/{len(ep_list)} — AUROC: {current_means}")

    results = {}
    summary_lines = []
    summary_lines.append(f"\n{'-'*65}")
    summary_lines.append(
        f"{'Task':<40} {'Acc':>10} {'F1':>10} {'AUROC':>10} {'AUPRC':>10}  (mean±std)"
    )
    summary_lines.append(f"{'-'*65}")

    for i, task_name in enumerate(task_names):
        auroc_mean = np.mean(exp[i])      if exp[i]          else float('nan')
        auroc_std  = np.std(exp[i])       if len(exp[i])     > 1 else 0.0
        auprc_mean = np.mean(auprc_exp[i])if auprc_exp[i]    else float('nan')
        auprc_std  = np.std(auprc_exp[i]) if len(auprc_exp[i])> 1 else 0.0
        acc_mean   = np.mean(acc_exp[i])  if acc_exp[i]      else float('nan')
        acc_std    = np.std(acc_exp[i])   if len(acc_exp[i]) > 1 else 0.0
        f1_mean    = np.mean(f1_exp[i])   if f1_exp[i]       else float('nan')
        f1_std     = np.std(f1_exp[i])    if len(f1_exp[i])  > 1 else 0.0

        line = (f"{task_name:<40} "
                f"{acc_mean:.4f}±{acc_std:.4f}  "
                f"{f1_mean:.4f}±{f1_std:.4f}  "
                f"{auroc_mean:.4f}±{auroc_std:.4f}  "
                f"{auprc_mean:.4f}±{auprc_std:.4f}")
        summary_lines.append(line)
        print(f"    {line}")

        results[task_name] = {
            'auroc':     auroc_mean,
            'auroc_std': auroc_std,
            'auprc':     auprc_mean,
            'auprc_std': auprc_std,
            'acc':       acc_mean,
            'acc_std':   acc_std,
            'f1':        f1_mean,
            'f1_std':    f1_std,
            'raw_auroc': exp[i],
            'raw_auprc': auprc_exp[i],
        }

    all_auroc = [results[t]['auroc'] for t in task_names if not np.isnan(results[t]['auroc'])]
    all_auprc = [results[t]['auprc'] for t in task_names if not np.isnan(results[t]['auprc'])]
    all_acc   = [results[t]['acc']   for t in task_names if not np.isnan(results[t]['acc'])]
    all_f1    = [results[t]['f1']    for t in task_names if not np.isnan(results[t]['f1'])]

    ov_acc   = np.mean(all_acc)   if all_acc   else float('nan')
    ov_f1    = np.mean(all_f1)    if all_f1    else float('nan')
    ov_auroc = np.mean(all_auroc) if all_auroc else float('nan')
    ov_auprc = np.mean(all_auprc) if all_auprc else float('nan')

    ov_line = (f"\n{'OVERALL AVERAGE':<40} "
               f"{ov_acc:.4f}{'':>14}"
               f"{ov_f1:.4f}{'':>14}"
               f"{ov_auroc:.4f}{'':>14}"
               f"{ov_auprc:.4f}")
    summary_lines.append(ov_line)
    summary_lines.append('-'*65)
    print(ov_line)

    with open(episode_file, 'a', encoding='utf-8') as rf:
        rf.write('\n'.join(summary_lines) + '\n')

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3Br-MGD branch ablations')
    parser.add_argument('--data_dir',        type=str, required=True)
    parser.add_argument('--checkpoint_dir',  type=str, default='checkpoints')
    parser.add_argument('--dataset',         type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',           type=int, nargs='+', default=[5, 10])
    parser.add_argument('--episodes_file',   type=str, default='baselines/episodes_seed42_tox21.json')
    parser.add_argument('--variants',        type=str, nargs='+', default=['all'],
                        help="List of variants to eval, e.g. 'all' or 'v1 v2 v3'")
    parser.add_argument('--seed',            type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'all' in args.variants:
        variants_to_run = list(VARIANT_INFO.keys())
    else:
        variants_to_run = [v for v in args.variants if v in VARIANT_INFO]

    if not variants_to_run:
        print("No valid variants found. Exit.")
        return

    print(f"Device   : {device}")
    print(f"Dataset  : {args.dataset}")
    print(f"Shots    : {args.shots}")
    print(f"Variants : {variants_to_run}")
    print(f"Seed     : {args.seed}")

    os.makedirs('results', exist_ok=True)

    _, meta_test = load_all_splits(args.data_dir)
    
    # Load episodes
    with open(args.episodes_file, 'r', encoding='utf-8') as f:
        episodes_data = json.load(f)

    # all_results[shot_name][variant] = {task: metrics}
    all_results = {}

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        all_results[shot_name] = {}

        if shot_name not in episodes_data:
            print(f"WARNING: {shot_name} not found in {args.episodes_file}. Skipping.")
            continue
            
        ep_list = episodes_data[shot_name]

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
            ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
            protonet = build_model(variant, device, vocab_size=SMILES_VOCAB.vocab_size)
            protonet.load_state_dict(ckpt['model_state'])
            print(f"     epoch={ckpt.get('epoch','?')}, "
                  f"val_auroc={ckpt.get('val_auroc', 0):.4f}")

            # Evaluate
            task_results = evaluate_branch(
                protonet      = protonet,
                ep_list       = ep_list,
                meta_test     = meta_test,
                device        = device,
                K_shot        = K_shot,
                variant       = variant,
                dataset       = args.dataset,
                shot_name     = shot_name,
                result_dir    = 'results',
            )

            all_results[shot_name][variant] = task_results

            aurocs = [r['auroc'] for r in task_results.values()
                      if not np.isnan(r['auroc'])]
            auprcs = [r['auprc'] for r in task_results.values()
                      if not np.isnan(r['auprc'])]
            if aurocs:
                print(f"     Mean AUROC = {np.mean(aurocs):.4f}")
            if auprcs:
                print(f"     Mean AUPRC = {np.mean(auprcs):.4f}")

    json_path = os.path.join('results', f'ablation_{args.dataset}_summary.json')
    json_data = {}
    for shot_name, variant_results in all_results.items():
        json_data[shot_name] = {}
        for variant, task_results in variant_results.items():
            aurocs = [r['auroc'] for r in task_results.values()
                      if not np.isnan(r['auroc'])]
            auprcs = [r['auprc'] for r in task_results.values()
                      if not np.isnan(r['auprc'])]
            json_data[shot_name][variant] = {
                'variant_name': VARIANT_INFO[variant]['name'],
                'mean_auroc':   float(np.mean(aurocs)) if aurocs else None,
                'mean_auprc':   float(np.mean(auprcs)) if auprcs else None,
                'per_task': {
                    t: {k: float(v) if not isinstance(v, list) else v
                        for k, v in r.items()
                        if k not in ['raw_auroc', 'raw_auprc']}
                    for t, r in task_results.items()
                },
            }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\nSummary JSON → {json_path}")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()