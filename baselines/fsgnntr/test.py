"""
test.py — Meta-testing for FS-GNNTR baseline.

Protocol:
  - Loads shared pre-generated episodes (episodes_seed42_<dataset>.json)
  - For each episode: MAML adaptation on support (k_test=20 steps), evaluate on query
  - Reports ROC-AUC mean ± std per task and overall
  - Saves results.json in shared schema
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from sklearn.metrics import roc_auc_score
from collections import defaultdict

# ── Path setup ──────────────────────────────────────────────────────────────────
_BASELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_BASELINE_ROOT, '..'))
_BRMGD_PATH    = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')

# Insert in REVERSE priority order (last insert = highest priority at sys.path[0])
# CRITICAL: BRMGD must come AFTER FSGNNTR so it takes priority; both have data.py
for p in [_BRMGD_PATH, _BASELINE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from seed_utils import set_seed
from episode_manager import load_episodes, reconstruct_sample_from_smiles
from graph_adapter import adapt_sample_to_fsgnntr
from maml_utils import build_fsgnntr_batch

from vendor.transformer import GNN_prediction, TR
import config as CFG
from train import (
    TR_dynamic, build_fsgnntr_model,
    forward_fsgnntr, compute_fsgnntr_losses, compute_fsgnntr_auroc,
    maml_inner_fsgnntr,
)


def meta_test_step_fsgnntr(
    gnn, tr, support_samples, query_samples, device,
    gnn_crit, tr_crit, lr_update, n_inner_steps
):
    """
    One FS-GNNTR test episode.
    Returns ROC-AUC (float or NaN).
    """
    support_adapted = [adapt_sample_to_fsgnntr(s) for s in support_samples if s is not None]
    query_adapted   = [adapt_sample_to_fsgnntr(q) for q in query_samples   if q is not None]

    if len(support_adapted) == 0 or len(query_adapted) == 0:
        return float('nan')

    support_batch = build_fsgnntr_batch(support_adapted, device)
    query_batch   = build_fsgnntr_batch(query_adapted, device)

    # Save original params
    orig_gnn = parameters_to_vector(gnn.parameters()).detach().clone()
    orig_tr  = parameters_to_vector(tr.parameters()).detach().clone()

    # MAML adaptation on support (n_inner_steps)
    gnn.train(); tr.train()
    with torch.enable_grad():
        for step in range(n_inner_steps):
            retain = (step < n_inner_steps - 1)
            gnn_loss, tr_loss = compute_fsgnntr_losses(gnn, tr, support_batch, gnn_crit, tr_crit)

            grads_gnn = torch.autograd.grad(
                gnn_loss, gnn.parameters(), retain_graph=True, allow_unused=True
            )
            g_gnn = parameters_to_vector([
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads_gnn, gnn.parameters())
            ])
            vector_to_parameters(
                parameters_to_vector(gnn.parameters()) - lr_update * g_gnn,
                gnn.parameters()
            )

            grads_tr = torch.autograd.grad(
                tr_loss, tr.parameters(), retain_graph=retain, allow_unused=True
            )
            g_tr = parameters_to_vector([
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads_tr, tr.parameters())
            ])
            vector_to_parameters(
                parameters_to_vector(tr.parameters()) - lr_update * g_tr,
                tr.parameters()
            )

    # Evaluate on query
    gnn.eval(); tr.eval()
    auroc = compute_fsgnntr_auroc(gnn, tr, query_batch)

    # Restore original params
    vector_to_parameters(orig_gnn, gnn.parameters())
    vector_to_parameters(orig_tr,  tr.parameters())

    return auroc


def evaluate(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {args.shots}")

    # Positive weight
    pos_weight = CFG.POS_WEIGHT_TOX21 if args.dataset == 'tox21' else CFG.POS_WEIGHT_SIDER

    # Load episodes
    episodes = load_episodes(args.episodes_file)
    print(f"Loaded episodes from: {args.episodes_file}")
    print(f"Meta-test tasks: {episodes['meta_test_tasks']}")

    from episode_manager import build_smiles_lookup
    build_smiles_lookup(args.data_dir)

    all_results = {}

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        if shot_name not in episodes:
            print(f"  WARNING: {shot_name} not found in episodes file. Skipping.")
            continue

        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")
        set_seed(args.seed)

        # Load model
        gnn, tr, gnn_crit, tr_crit = build_fsgnntr_model(device, args.pretrained, pos_weight)

        ckpt_path = args.checkpoint or os.path.join(
            CFG.CHECKPOINT_DIR, f"fsgnntr_{args.dataset}_{shot_name}_best.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            gnn.load_state_dict(ckpt['gnn_state'])
            tr.load_state_dict(ckpt['tr_state'])
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: No checkpoint found at {ckpt_path}. Using pretrained GNN only.")

        task_results = defaultdict(list)
        ep_list = episodes[shot_name]

        from tqdm import tqdm
        for ep in tqdm(ep_list, desc=f"Evaluating {shot_name}"):
            task_name = ep['task']

            support_samples = [
                reconstruct_sample_from_smiles(sm, lb)
                for sm, lb in zip(ep['support_smiles'], ep['support_labels'])
            ]
            query_samples = [
                reconstruct_sample_from_smiles(sm, lb)
                for sm, lb in zip(ep['query_smiles'], ep['query_labels'])
            ]

            # Filter None (invalid SMILES)
            support_samples = [s for s in support_samples if s is not None]
            query_samples   = [q for q in query_samples   if q is not None]

            auroc = meta_test_step_fsgnntr(
                gnn, tr, support_samples, query_samples, device,
                gnn_crit, tr_crit,
                lr_update=CFG.LR_UPDATE, n_inner_steps=CFG.N_INNER_TEST
            )

            if not np.isnan(auroc):
                task_results[task_name].append(auroc)

        # Compute per-task statistics
        per_task = {}
        all_task_means = []
        for task_name, aucs in task_results.items():
            mean_auc = float(np.mean(aucs))
            std_auc  = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            per_task[task_name] = {
                'auc_mean': mean_auc,
                'auc_std':  std_auc,
                'n_episodes': len(aucs),
                'raw_auc': [round(a, 4) for a in aucs],
            }
            all_task_means.append(mean_auc)
            print(f"  {task_name:40s}: AUC = {mean_auc:.4f} ± {std_auc:.4f}  (n={len(aucs)})")

        overall_mean = float(np.mean(all_task_means)) if all_task_means else float('nan')
        overall_std  = float(np.std(all_task_means, ddof=1)) if len(all_task_means) > 1 else 0.0
        print(f"  {'Overall':40s}: AUC = {overall_mean:.4f} ± {overall_std:.4f}")

        all_results[shot_name] = {
            'auc_mean':   overall_mean,
            'auc_std':    overall_std,
            'per_task':   per_task,
        }

    # Save results.json
    results = {
        'model':            'fsgnntr',
        'dataset':          args.dataset,
        'gnn_type':         CFG.GNN_TYPE,
        'seed':             args.seed,
        'n_inner_test':     CFG.N_INNER_TEST,
        'meta_test_tasks':  episodes.get('meta_test_tasks', []),
        'shots':            all_results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-test FS-GNNTR baseline')
    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--output_dir',    type=str, default=CFG.CHECKPOINT_DIR)
    parser.add_argument('--dataset',       type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',         type=int, nargs='+', default=[5, 10])
    parser.add_argument('--pretrained',    type=str, default=CFG.PRETRAINED)
    parser.add_argument('--checkpoint',    type=str, default=None,
                        help='Path to checkpoint .pt file (overrides default)')
    parser.add_argument('--episodes_file', type=str, required=True,
                        help='Path to pre-generated episodes JSON')
    parser.add_argument('--seed',          type=int, default=CFG.SEED)
    args = parser.parse_args()
    evaluate(args)
