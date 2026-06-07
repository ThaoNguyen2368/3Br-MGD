import os
import json
import random
import argparse
import numpy as np
import torch

import sys
# Thêm thư mục chứa script này vào path để import local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_all_splits
from BrMGD_model import TripleEncoder, EnhancedProtoNet
from BrMGD_eval import evaluate_meta_task
from BrMGD_train import create_meta_task


def set_seed(seed: int = 42):
    """Fix tất cả nguồn ngẫu nhiên để đảm bảo test episodes reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_meta_tasks(
    protonet,
    ep_list: list,
    meta_test: dict,
    device,
    K_shot: int,
    result_file: str,
) -> dict:
    
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
    
    # Map task names to indices for easy storage
    task_to_idx = {t: i for i, t in enumerate(task_names)}

    exp = [[] for _ in range(n_tasks)]
    auprc_exp = [[] for _ in range(n_tasks)]
    acc_exp  = [[] for _ in range(n_tasks)]
    f1_exp   = [[] for _ in range(n_tasks)]

    WRITE_LIMIT = 30  

    os.makedirs(os.path.dirname(result_file) if os.path.dirname(result_file) else '.', exist_ok=True)

    episodes = len(ep_list) // n_tasks if n_tasks > 0 else 0

    with open(result_file, 'a', encoding='utf-8') as rf:
        rf.write(f"\n{'='*60}\n")
        rf.write(f"Tasks: {task_names}\n")
        rf.write(f"K_shot={K_shot}, Total episodes={len(ep_list)}\n")
        rf.write(f"{'='*60}\n")

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
            acc, f1, auroc, auprc = evaluate_meta_task(protonet, support, query, device)
            exp[i].append(round(auroc, 4) if not np.isnan(auroc) else 0.0)
            auprc_exp[i].append(round(auprc, 4) if not np.isnan(auprc) else 0.0)
            if not np.isnan(acc):  acc_exp[i].append(acc)
            if not np.isnan(f1):   f1_exp[i].append(f1)
        except Exception as e:
            exp[i].append(0.0)
            auprc_exp[i].append(0.0)

        if ep_idx < WRITE_LIMIT * n_tasks:
            with open(result_file, 'a', encoding='utf-8') as rf:
                rf.write(f"Task {task_name} AUROC: \t{exp[i][-1] if exp[i] else 0.0} \tAUPRC: \t{auprc_exp[i][-1] if auprc_exp[i] else 0.0}\n")

        if (ep_idx + 1) % (10 * n_tasks) == 0:
            current_means = [
                f"{np.mean(exp[idx]):.4f}" if exp[idx] else 'nan'
                for idx in range(n_tasks)
            ]
            print(f"  Processed {ep_idx + 1}/{len(ep_list)} — AUROC: {current_means}")

    results = {}
    summary_lines = []
    summary_lines.append(f"\n{'-'*60}")
    summary_lines.append(f"{'Task':<40} {'Acc':>8} {'F1':>8} {'AUROC':>8} {'AUPRC':>8}  (mean ± std)")
    summary_lines.append(f"{'-'*60}")

    for i, task_name in enumerate(task_names):
        auroc_scores = exp[i]
        auprc_scores = auprc_exp[i]
        acc_scores   = acc_exp[i]
        f1_scores    = f1_exp[i]

        auroc_mean = np.mean(auroc_scores) if auroc_scores else float('nan')
        auroc_std  = np.std(auroc_scores)  if len(auroc_scores) > 1 else 0.0
        auprc_mean = np.mean(auprc_scores) if auprc_scores else float('nan')
        auprc_std  = np.std(auprc_scores)  if len(auprc_scores) > 1 else 0.0
        acc_mean   = np.mean(acc_scores)   if acc_scores   else float('nan')
        acc_std    = np.std(acc_scores)    if len(acc_scores)   > 1 else 0.0
        f1_mean    = np.mean(f1_scores)    if f1_scores    else float('nan')
        f1_std     = np.std(f1_scores)     if len(f1_scores)    > 1 else 0.0

        line = (f"{task_name:<40} "
                f"{acc_mean:.4f}±{acc_std:.4f}  "
                f"{f1_mean:.4f}±{f1_std:.4f}  "
                f"{auroc_mean:.4f}±{auroc_std:.4f}  "
                f"{auprc_mean:.4f}±{auprc_std:.4f}")
        summary_lines.append(line)
        print(f"  {line}")

        results[task_name] = {
            'auroc':           auroc_mean,
            'auroc_std':       auroc_std,
            'auprc':           auprc_mean,
            'auprc_std':       auprc_std,
            'acc':             acc_mean,
            'acc_std':         acc_std,
            'f1':              f1_mean,
            'f1_std':          f1_std,
            'raw_auroc_scores': auroc_scores,
            'raw_auprc_scores': auprc_scores,
        }

    # Overall average
    all_auroc = [results[t]['auroc'] for t in task_names if not np.isnan(results[t]['auroc'])]
    all_auprc = [results[t]['auprc'] for t in task_names if not np.isnan(results[t]['auprc'])]
    all_acc   = [results[t]['acc']   for t in task_names if not np.isnan(results[t]['acc'])]
    all_f1    = [results[t]['f1']    for t in task_names if not np.isnan(results[t]['f1'])]

    ov_line = (f"\n{'OVERALL AVERAGE':<40} "
               f"{np.mean(all_acc):.4f}±{np.std(all_acc):.4f}  "
               f"{np.mean(all_f1):.4f}±{np.std(all_f1):.4f}  "
               f"{np.mean(all_auroc):.4f}±{np.std(all_auroc):.4f}  "
               f"{np.mean(all_auprc):.4f}±{np.std(all_auprc):.4f}")
    summary_lines.append(ov_line)
    summary_lines.append('-'*60)
    print(ov_line)


    with open(result_file, 'a', encoding='utf-8') as rf:
        rf.write('\n'.join(summary_lines) + '\n')

    return results


def aggregate_results(results_per_shot: dict) -> dict:
    aggregated = {}
    for shot_name, task_results in results_per_shot.items():
        aurocs = [r['auroc'] for r in task_results.values() if not np.isnan(r['auroc'])]
        auprcs = [r['auprc'] for r in task_results.values() if not np.isnan(r['auprc'])]
        accs   = [r['acc']   for r in task_results.values() if not np.isnan(r['acc'])]
        f1s    = [r['f1']    for r in task_results.values() if not np.isnan(r['f1'])]

        aggregated[shot_name] = {
            'task_results':    task_results,
            'overall_auroc':   np.mean(aurocs) if aurocs else float('nan'),
            'overall_auroc_std': np.std(aurocs) if aurocs else 0.0,
            'overall_auprc':   np.mean(auprcs) if auprcs else float('nan'),
            'overall_auprc_std': np.std(auprcs) if auprcs else 0.0,
            'overall_acc':     np.mean(accs)   if accs   else float('nan'),
            'overall_f1':      np.mean(f1s)    if f1s    else float('nan'),
            'n_tasks':         len(task_results),
        }
    return aggregated


def print_summary_table(aggregated: dict, dataset: str, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)

    for shot_name, agg in aggregated.items():
        shot_num  = shot_name.replace('-shot', '')
        file_name = f"mean-3BrMGD_{dataset}_{shot_num}shot.txt"
        file_path = os.path.join(result_dir, file_name)

        lines = []
        lines.append(f"3BrMGD — {dataset.upper()} — {shot_name}")
        lines.append('='*70)
        lines.append(f"{'Task':<45} {'Acc':>10} {'F1':>10} {'AUROC':>10} {'AUPRC':>10}")
        lines.append('-'*70)

        task_results = agg['task_results']
        for task_name, r in task_results.items():
            lines.append(
                f"{task_name:<45} "
                f"{r['acc']:.4f}±{r['acc_std']:.4f}  "
                f"{r['f1']:.4f}±{r['f1_std']:.4f}  "
                f"{r['auroc']:.4f}±{r['auroc_std']:.4f}  "
                f"{r['auprc']:.4f}±{r['auprc_std']:.4f}"
            )

        lines.append('-'*70)
        lines.append(
            f"{'OVERALL AVERAGE':<45} "
            f"{agg['overall_acc']:.4f}{'':>10}"
            f"{agg['overall_f1']:.4f}{'':>10}"
            f"{agg['overall_auroc']:.4f}±{agg['overall_auroc_std']:.4f}  "
            f"{agg['overall_auprc']:.4f}±{agg['overall_auprc_std']:.4f}"
        )
        lines.append('='*70)

        output = '\n'.join(lines)
        print(f"\n{output}")

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output + '\n')
        print(f"Summary saved → {file_path}")

    if '5-shot' in aggregated and '10-shot' in aggregated:
        print("\n" + "="*50)
        print("5-SHOT vs 10-SHOT COMPARISON")
        print("="*50)
        for sname in ['5-shot', '10-shot']:
            agg = aggregated[sname]
            print(f"  {sname:10s}: AUROC = {agg['overall_auroc']:.4f}, AUPRC = {agg['overall_auprc']:.4f} "
                  f"(avg over {agg['n_tasks']} tasks)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3BRMGD on meta-test tasks')
    parser.add_argument('--data_dir',        type=str, required=True)
    parser.add_argument('--checkpoint_dir',  type=str, default='checkpoints')
    parser.add_argument('--dataset',         type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',           type=int, nargs='+', default=[5, 10])
    parser.add_argument('--episodes_file',   type=str, default='baselines/episodes_seed42_tox21.json')
    parser.add_argument('--seed',            type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {args.shots}")
    print(f"Seed    : {args.seed}")

    os.makedirs('results', exist_ok=True)

    # Load data
    _, meta_test = load_all_splits(args.data_dir)
    
    # Load episodes
    with open(args.episodes_file, 'r', encoding='utf-8') as f:
        episodes_data = json.load(f)

    results_per_shot = {}

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"BrMGD_{args.dataset}_{shot_name}_best.pth"
        )

        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint is not exist: {ckpt_path}. Skipping.")
            continue

        if shot_name not in episodes_data:
            print(f"WARNING: {shot_name} not found in {args.episodes_file}. Skipping.")
            continue
            
        ep_list = episodes_data[shot_name]

        print(f"\n{'='*25} {shot_name.upper()} {'='*25}")
        print(f"Loading checkpoint: {ckpt_path}")

        # Rebuild model
        encoder  = TripleEncoder().to(device)
        protonet = EnhancedProtoNet(encoder).to(device)

        try:
            # Load file .pth
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Kiểm tra xem ckpt là dict hay chỉ là state_dict thuần túy
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                protonet.load_state_dict(ckpt['model_state'])
                epoch_info = ckpt.get('epoch', '?')
                val_info = ckpt.get('val_auroc', 0.0)
            else:
                # Trường hợp file .pth chỉ lưu mỗi state_dict
                protonet.load_state_dict(ckpt)
                epoch_info = "Unknown"
                val_info = 0.0
                
            print(f"✅ Loaded: Epoch={epoch_info}, Train_Query_AUROC={val_info:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading {ckpt_path}: {e}")
            continue
        # Result file
        shot_num     = shot_name.replace('-shot', '')
        result_file  = os.path.join('results', f"mean-3BrMGD_{args.dataset}_{shot_num}shot.txt")

        print(f"  Running {len(ep_list)} test episodes from {args.episodes_file}...")
        test_results = evaluate_meta_tasks(
            protonet, ep_list, meta_test, device,
            K_shot, result_file=result_file,
        )

        results_per_shot[shot_name] = test_results

    if not results_per_shot:
        print("No result, recheck --checkpoint_dir.")
        return

    aggregated = aggregate_results(results_per_shot)

    json_path = os.path.join('results', 'final_results.json')
    json_data = {}
    for shot_name, agg in aggregated.items():
        json_data[shot_name] = {
            'overall_auroc':     agg['overall_auroc'],
            'overall_auroc_std': agg['overall_auroc_std'],
            'overall_auprc':     agg['overall_auprc'],
            'overall_auprc_std': agg['overall_auprc_std'],
            'overall_acc':       agg['overall_acc'],
            'overall_f1':        agg['overall_f1'],
            'n_tasks':           agg['n_tasks'],
            'per_task': {
                task: {k: v for k, v in r.items() if k not in ['raw_auroc_scores', 'raw_auprc_scores']}
                for task, r in agg['task_results'].items()
            }
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\nFinal results JSON → {json_path}")


if __name__ == '__main__':
    main()