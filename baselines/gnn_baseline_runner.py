"""
gnn_baseline_runner.py — Shared train/test runner for GNN-only baselines.

Used by: GCN, GIN, GraphSAGE
Each baseline imports run_train() and run_test() with its specific config module.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from collections import defaultdict
from copy import deepcopy

# ── Path setup ──────────────────────────────────────────────────────────────────
_BASELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_BASELINE_ROOT, '..'))
_BRMGD_PATH    = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')

# Insert in REVERSE priority order (last insert = highest priority at sys.path[0])
# CRITICAL: BRMGD must come AFTER FSGNNTR in this loop so it ends up first in sys.path.
# Both repos have a 'data.py'; we want 3Br-MGD's data.py (has load_all_splits) to win.
for p in [_BRMGD_PATH, _BASELINE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from seed_utils import set_seed
from data import load_all_splits
from BrMGD_train import create_meta_task
from graph_adapter import adapt_sample_to_fsgnntr
from maml_utils import (
    build_fsgnntr_batch, compute_gnn_loss, compute_gnn_auroc,
    maml_inner_update, meta_train_step_gnn, meta_test_step_gnn,
)
from episode_manager import load_episodes, reconstruct_sample_from_smiles

from vendor.transformer import GNN_prediction


# ─── Model builder ───────────────────────────────────────────────────────────────

def build_gnn_model(CFG, device):
    """
    Build GNN_prediction and load pretrained weights.
    baseline=1 mode: GNN only, no Transformer.
    """
    gnn = GNN_prediction(
        CFG.GRAPH_LAYERS, CFG.EMB_SIZE,
        jk=CFG.JK, dropout_prob=CFG.DROPOUT,
        pooling=CFG.POOLING, gnn_type=CFG.GNN_TYPE
    ).to(device)

    # state = torch.load(CFG.PRETRAINED, map_location=device, weights_only=False)
    # gnn.gnn.load_state_dict(state, strict=False)
    # print(f"  Loaded pretrained GNN [{CFG.GNN_TYPE}] from: {CFG.PRETRAINED}")
    print(f"  Randomly initialized GNN [{CFG.GNN_TYPE}] (pretrained weights disabled)")

    criterion = nn.BCEWithLogitsLoss()  # baseline=1 uses no pos_weight
    return gnn, criterion


# ─── Training ───────────────────────────────────────────────────────────────────

def run_train(CFG, args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Model   : {CFG.MODEL_NAME}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {args.shots}")

    os.makedirs(args.output_dir, exist_ok=True)

    meta_train, _ = load_all_splits(args.data_dir)
    task_names = list(meta_train.keys())
    print(f"Meta-train tasks ({len(task_names)}): {task_names}")

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")

        set_seed(args.seed)

        gnn, criterion = build_gnn_model(CFG, device)

        optimizer = torch.optim.Adam(
            [
                {'params': gnn.gnn.parameters()},
                {'params': gnn.graph_pred_linear.parameters(), 'lr': CFG.LR_GNN},
            ],
            lr=CFG.LR_GNN, weight_decay=0
        )

        best_auroc   = 0.0
        best_state   = None
        patience_ctr = 0

        for epoch in range(1, args.max_epochs + 1):
            epoch_aurocs = []
            optimizer.zero_grad()

            for ep_idx in range(args.train_episodes):
                task_name = random.choice(task_names)
                task_data = meta_train[task_name]

                _, auroc = meta_train_step_gnn(
                    model             = gnn,
                    optimizer         = optimizer,
                    task_data         = task_data,
                    K_shot            = K_shot,
                    Q_query           = args.q_query,
                    device            = device,
                    criterion         = criterion,
                    lr_update         = CFG.LR_UPDATE,
                    n_inner_steps     = CFG.N_INNER_TRAIN,
                    create_meta_task_fn = create_meta_task,
                    adapt_sample_fn   = adapt_sample_to_fsgnntr,
                )

                if not np.isnan(auroc):
                    epoch_aurocs.append(auroc)

                # Per-episode meta-update
                optimizer.step()
                optimizer.zero_grad()

            epoch_auroc = np.mean(epoch_aurocs) if epoch_aurocs else float('nan')

            if not np.isnan(epoch_auroc) and epoch_auroc > (best_auroc + 0.001):
                best_auroc   = epoch_auroc
                best_state   = deepcopy(gnn.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: query_auc={epoch_auroc:.4f}, "
                      f"patience={patience_ctr}/{args.patience}")

            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}. Best AUC = {best_auroc:.4f}")
                break

        # Save
        if best_state is not None:
            ckpt = {
                'model_state': best_state,
                'val_auroc':   best_auroc,
                'config':      vars(args),
            }
            ckpt_path = os.path.join(
                args.output_dir,
                f"{CFG.MODEL_NAME.lower()}_{args.dataset}_{shot_name}_best.pt"
            )
            torch.save(ckpt, ckpt_path)
            print(f"  Best checkpoint → {ckpt_path}")

        last_ckpt = {'model_state': gnn.state_dict(), 'config': vars(args)}
        last_path = os.path.join(
            args.output_dir,
            f"{CFG.MODEL_NAME.lower()}_{args.dataset}_{shot_name}_last.pt"
        )
        torch.save(last_ckpt, last_path)

    print(f"\n{CFG.MODEL_NAME} training complete!")


# ─── Testing ─────────────────────────────────────────────────────────────────────

def run_test(CFG, args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Model   : {CFG.MODEL_NAME}")
    print(f"Dataset : {args.dataset}")

    episodes = load_episodes(args.episodes_file)
    print(f"Loaded episodes from: {args.episodes_file}")

    from episode_manager import build_smiles_lookup
    build_smiles_lookup(args.data_dir)

    gnn, criterion = build_gnn_model(CFG, device)

    all_results = {}

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        if shot_name not in episodes:
            print(f"  WARNING: {shot_name} not found in episodes file. Skipping.")
            continue

        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")
        set_seed(args.seed)

        # Load checkpoint
        ckpt_path = args.checkpoint or os.path.join(
            CFG.CHECKPOINT_DIR,
            f"{CFG.MODEL_NAME.lower()}_{args.dataset}_{shot_name}_best.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            gnn.load_state_dict(ckpt['model_state'])
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: No checkpoint found. Using randomly initialized GNN only.")

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
            support_samples = [s for s in support_samples if s is not None]
            query_samples   = [q for q in query_samples   if q is not None]

            auroc = meta_test_step_gnn(
                model         = gnn,
                support_samples = support_samples,
                query_samples   = query_samples,
                device          = device,
                criterion       = criterion,
                lr_update       = CFG.LR_UPDATE,
                n_inner_steps   = CFG.N_INNER_TEST,
                adapt_sample_fn = adapt_sample_to_fsgnntr,
            )

            if not np.isnan(auroc):
                task_results[task_name].append(auroc)

        per_task = {}
        all_task_means = []
        for task_name, aucs in task_results.items():
            m = float(np.mean(aucs))
            s = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            per_task[task_name] = {
                'auc_mean': m, 'auc_std': s,
                'n_episodes': len(aucs),
                'raw_auc': [round(a, 4) for a in aucs],
            }
            all_task_means.append(m)
            print(f"  {task_name:40s}: AUC = {m:.4f} ± {s:.4f}  (n={len(aucs)})")

        om = float(np.mean(all_task_means)) if all_task_means else float('nan')
        os_ = float(np.std(all_task_means, ddof=1)) if len(all_task_means) > 1 else 0.0
        print(f"  {'Overall':40s}: AUC = {om:.4f} ± {os_:.4f}")

        all_results[shot_name] = {'auc_mean': om, 'auc_std': os_, 'per_task': per_task}

    results = {
        'model':           CFG.MODEL_NAME,
        'dataset':         args.dataset,
        'gnn_type':        CFG.GNN_TYPE,
        'seed':            args.seed,
        'n_inner_test':    CFG.N_INNER_TEST,
        'meta_test_tasks': episodes.get('meta_test_tasks', []),
        'shots':           all_results,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")
