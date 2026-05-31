"""
train.py — Meta-training for FS-GNNTR baseline.

Protocol (matches 3Br-MGD for fairness):
  - Uses 3Br-MGD meta_train task split
  - Episode sampling via create_meta_task()
  - FOMAML with n_inner_steps inner updates per episode
  - 200 epochs max, patience=20 early stopping based on query AUC
  - Seed fixed at 42

Architecture (unchanged from original paper):
  GNN_prediction (5-layer GIN, emb=300) + TR (Vision Transformer, patched) + MAML

Key change from original:
  - TR.forward() reshape is dynamic (not hardcoded 10) via subclassing
  - Dataset/sampler replaced with 3Br-MGD's load_all_splits + create_meta_task
  - Task split replaced with 3Br-MGD TOX21_SPLITS / SIDER_SPLITS
"""

import os
import sys
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from sklearn.metrics import roc_auc_score
from copy import deepcopy

# ── Path setup ──────────────────────────────────────────────────────────────────
_BASELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_BASELINE_ROOT, '..'))
_BRMGD_PATH    = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')
_FSGNNTR_PATH  = os.path.join(_PROJECT_ROOT, 'FS-GNNTR_repo', 'FS-GNNTR')

# Insert in REVERSE priority order (last insert = highest priority at sys.path[0])
# CRITICAL: BRMGD must come AFTER FSGNNTR so it takes priority; both have data.py
for p in [_FSGNNTR_PATH, _BRMGD_PATH, _BASELINE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports ─────────────────────────────────────────────────────────────────────
from seed_utils import set_seed
from data import load_all_splits
from BrMGD_train import create_meta_task
from graph_adapter import adapt_sample_to_fsgnntr
from maml_utils import build_fsgnntr_batch, compute_gnn_loss

# FS-GNNTR original architecture (unchanged)
from transformer import GNN_prediction, TR
import config as CFG


# ─── Dynamic TR (fixes hardcoded batch_size=10 in TR.forward) ──────────────────

class TR_dynamic(TR):
    """
    Subclass of TR that uses dynamic batch size instead of hardcoded 10.
    Only the reshape line is changed. All other architecture is identical.
    Original: emb.reshape(10, 1, 300, 1)
    Fixed:    emb.reshape(emb.shape[0], 1, 300, 1)
    """
    def forward(self, emb):
        from einops import rearrange as _rearrange
        from transformer import position_emb as _position_emb
        batch_size = emb.shape[0]
        emb = emb.reshape(batch_size, 1, CFG.TR_EMB_SIZE, 1)  # dynamic (was hardcoded 10)
        h = self.to_patch_embedding(emb)
        pe = _position_emb(h)
        h = _rearrange(h, 'b ... d -> b (...) d') + pe
        h = self.transformer(h)
        h = h.mean(dim=1)
        h = self.to_latent(h)
        return self.linear_head(h), h


# ─── Model builder ──────────────────────────────────────────────────────────────

def build_fsgnntr_model(device, pretrained_path: str, pos_weight: float):
    """Build GNN_prediction + TR_dynamic models, load pretrained GNN weights."""
    gnn = GNN_prediction(
        CFG.GRAPH_LAYERS, CFG.EMB_SIZE,
        jk=CFG.JK, dropout_prob=CFG.DROPOUT,
        pooling=CFG.POOLING, gnn_type=CFG.GNN_TYPE
    ).to(device)

    tr = TR_dynamic(
        CFG.TR_EMB_SIZE, CFG.TR_PATCH_SIZE, CFG.TR_NUM_CLS,
        CFG.TR_DIM, CFG.TR_DEPTH, CFG.TR_HEADS, CFG.TR_MLP_DIM
    ).to(device)

    # Load pretrained GNN (strict=False: only loads matching layers)
    state = torch.load(pretrained_path, map_location=device)
    gnn.gnn.load_state_dict(state, strict=False)
    print(f"  Loaded pretrained GNN from: {pretrained_path}")

    gnn_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([pos_weight]).to(device)
    )
    tr_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.FloatTensor([pos_weight]).to(device)
    )

    return gnn, tr, gnn_criterion, tr_criterion


# ─── MAML forward (GNN + TR) ────────────────────────────────────────────────────

def forward_fsgnntr(gnn, tr, batch):
    """
    Forward pass through GNN + TR.
    Returns: gnn_pred [B,1], tr_pred [B,1], tr_emb [B,128]
    """
    gnn_pred, node_emb = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    graph_emb = gnn.pool(node_emb, batch.batch)   # [B, EMB_SIZE]
    tr_pred, tr_emb = tr(graph_emb)               # [B, 1], [B, 128]
    return gnn_pred, tr_pred, tr_emb


def compute_fsgnntr_losses(gnn, tr, batch, gnn_crit, tr_crit):
    """Returns (gnn_loss, tr_loss) scalars."""
    gnn_pred, tr_pred, _ = forward_fsgnntr(gnn, tr, batch)
    y = batch.y.view(gnn_pred.shape).to(torch.float64)
    gnn_loss = gnn_crit(gnn_pred.double(), y).sum() / gnn_pred.shape[0]
    tr_loss  = tr_crit(F.sigmoid(tr_pred).double(), y).sum() / tr_pred.shape[0]
    return gnn_loss, tr_loss


def compute_fsgnntr_auroc(gnn, tr, batch):
    """Compute ROC-AUC using Transformer output probabilities."""
    with torch.no_grad():
        _, tr_pred, _ = forward_fsgnntr(gnn, tr, batch)
        proba = torch.sigmoid(tr_pred).squeeze(1).cpu().numpy()
        y_true = batch.y.view(-1).cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return float('nan')


# ─── MAML inner update (GNN + TR) ───────────────────────────────────────────────

def maml_inner_fsgnntr(gnn, tr, support_batch, gnn_crit, tr_crit, lr_update, n_steps=1):
    """
    FOMAML inner update for FS-GNNTR (both GNN and TR).
    Returns (original_gnn_params, original_tr_params) for restoration.
    """
    orig_gnn = parameters_to_vector(gnn.parameters()).detach().clone()
    orig_tr  = parameters_to_vector(tr.parameters()).detach().clone()

    for step in range(n_steps):
        retain = (step < n_steps - 1)
        gnn_loss, tr_loss = compute_fsgnntr_losses(gnn, tr, support_batch, gnn_crit, tr_crit)

        # GNN inner update
        grads_gnn = torch.autograd.grad(
            gnn_loss, gnn.parameters(), retain_graph=True, allow_unused=True
        )
        g_gnn = parameters_to_vector([
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads_gnn, gnn.parameters())
        ])
        updated_gnn = parameters_to_vector(gnn.parameters()) - lr_update * g_gnn
        vector_to_parameters(updated_gnn, gnn.parameters())

        # TR inner update
        grads_tr = torch.autograd.grad(
            tr_loss, tr.parameters(), retain_graph=retain, allow_unused=True
        )
        g_tr = parameters_to_vector([
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads_tr, tr.parameters())
        ])
        updated_tr = parameters_to_vector(tr.parameters()) - lr_update * g_tr
        vector_to_parameters(updated_tr, tr.parameters())

    return orig_gnn, orig_tr


# ─── One meta-train episode ─────────────────────────────────────────────────────

def meta_train_step_fsgnntr(
    gnn, tr, task_data, K_shot, Q_query, device,
    gnn_crit, tr_crit, lr_update, n_inner_steps
):
    """
    One FOMAML training episode for FS-GNNTR.
    Accumulates gradients in .grad — caller must zero_grad before and step after.
    Returns (query_auroc: float)
    """
    try:
        support, query = create_meta_task(task_data, K_shot, Q_query, train=True)
    except ValueError:
        return float('nan')

    support_adapted = [adapt_sample_to_fsgnntr(s) for s in support]
    query_adapted   = [adapt_sample_to_fsgnntr(q) for q in query]

    if len(support_adapted) == 0 or len(query_adapted) == 0:
        return float('nan')

    support_batch = build_fsgnntr_batch(support_adapted, device)
    query_batch   = build_fsgnntr_batch(query_adapted, device)

    gnn.train(); tr.train()

    # Inner update
    orig_gnn, orig_tr = maml_inner_fsgnntr(
        gnn, tr, support_batch, gnn_crit, tr_crit, lr_update, n_inner_steps
    )

    # Query loss (with adapted params)
    gnn_query_loss, tr_query_loss = compute_fsgnntr_losses(
        gnn, tr, query_batch, gnn_crit, tr_crit
    )

    # Monitoring AUC
    gnn.eval(); tr.eval()
    query_auroc = compute_fsgnntr_auroc(gnn, tr, query_batch)
    gnn.train(); tr.train()

    # Accumulate meta-gradients
    gnn_query_loss.backward(retain_graph=True)
    tr_query_loss.backward()

    # Restore
    vector_to_parameters(orig_gnn, gnn.parameters())
    vector_to_parameters(orig_tr,  tr.parameters())

    return query_auroc


# ─── Training loop ───────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {args.shots}")
    print(f"Seed    : {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    meta_train, _ = load_all_splits(args.data_dir)
    task_names = list(meta_train.keys())
    print(f"Meta-train tasks ({len(task_names)}): {task_names}")

    # Positive weight per dataset
    pos_weight = CFG.POS_WEIGHT_TOX21 if args.dataset == 'tox21' else CFG.POS_WEIGHT_SIDER

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")

        set_seed(args.seed)  # re-seed for each shot

        gnn, tr, gnn_crit, tr_crit = build_fsgnntr_model(device, args.pretrained, pos_weight)

        opt_gnn = torch.optim.Adam([
            {'params': gnn.gnn.parameters()},
            {'params': gnn.graph_pred_linear.parameters(), 'lr': CFG.LR_GNN},
        ], lr=CFG.LR_GNN, weight_decay=0)
        opt_tr = torch.optim.Adam(tr.parameters(), lr=CFG.LR_TR)

        best_auroc      = 0.0
        best_gnn_state  = None
        best_tr_state   = None
        patience_count  = 0

        for epoch in range(1, args.max_epochs + 1):
            epoch_aurocs = []

            opt_gnn.zero_grad()
            opt_tr.zero_grad()

            for ep_idx in range(args.train_episodes):
                task_name = random.choice(task_names)
                task_data = meta_train[task_name]

                auroc = meta_train_step_fsgnntr(
                    gnn, tr, task_data, K_shot, args.q_query, device,
                    gnn_crit, tr_crit, CFG.LR_UPDATE, CFG.N_INNER_TRAIN
                )
                if not np.isnan(auroc):
                    epoch_aurocs.append(auroc)

                # Per-episode meta-update (consistent with 3Br-MGD style)
                opt_gnn.step(); opt_gnn.zero_grad()
                opt_tr.step();  opt_tr.zero_grad()

            epoch_auroc = np.mean(epoch_aurocs) if epoch_aurocs else float('nan')

            # Early stopping
            if not np.isnan(epoch_auroc) and epoch_auroc > (best_auroc + 0.001):
                best_auroc     = epoch_auroc
                best_gnn_state = deepcopy(gnn.state_dict())
                best_tr_state  = deepcopy(tr.state_dict())
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: query_auc={epoch_auroc:.4f}, "
                      f"patience={patience_count}/{args.patience}")

            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch}. Best AUC = {best_auroc:.4f}")
                break

        # Save checkpoints
        if best_gnn_state is not None:
            ckpt = {
                'gnn_state':  best_gnn_state,
                'tr_state':   best_tr_state,
                'val_auroc':  best_auroc,
                'config':     vars(args),
            }
            ckpt_path = os.path.join(args.output_dir, f"fsgnntr_{args.dataset}_{shot_name}_best.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  Best checkpoint → {ckpt_path}")

        last_ckpt = {
            'gnn_state': gnn.state_dict(),
            'tr_state':  tr.state_dict(),
            'config':    vars(args),
        }
        last_path = os.path.join(args.output_dir, f"fsgnntr_{args.dataset}_{shot_name}_last.pt")
        torch.save(last_ckpt, last_path)

    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-train FS-GNNTR baseline')
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default=CFG.CHECKPOINT_DIR)
    parser.add_argument('--dataset',        type=str, default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',          type=int, nargs='+', default=[5, 10])
    parser.add_argument('--max_epochs',     type=int, default=CFG.MAX_EPOCHS)
    parser.add_argument('--patience',       type=int, default=CFG.PATIENCE)
    parser.add_argument('--train_episodes', type=int, default=CFG.TRAIN_EPISODES)
    parser.add_argument('--q_query',        type=int, default=CFG.Q_QUERY)
    parser.add_argument('--pretrained',     type=str, default=CFG.PRETRAINED)
    parser.add_argument('--seed',           type=int, default=CFG.SEED)
    args = parser.parse_args()
    train(args)
