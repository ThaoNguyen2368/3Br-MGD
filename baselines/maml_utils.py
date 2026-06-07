"""
maml_utils.py — Shared FOMAML utilities for GNN-only baselines (GCN, GIN, GraphSAGE).

All GNN-only baselines share the same MAML training/testing logic.
FS-GNNTR has its own train/test because it also updates the Transformer.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch, Data
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from sklearn.metrics import roc_auc_score


# ─── Batch builders ─────────────────────────────────────────────────

def build_fsgnntr_batch(adapted_samples: list, device: torch.device, label_dtype=torch.float):
    """
    Collate adapted FS-GNNTR/GCN/GIN/GraphSAGE samples into a PyG Batch.

    Args:
        adapted_samples: list of {'graph': Data(x=long,edge_attr=long), 'label': int}
        device: torch.device

    Returns:
        batch: PyG Batch with batch.y = [B] float
    """
    graph_list = []
    labels = []
    for s in adapted_samples:
        g = s['graph']
        g.y = torch.tensor([s['label']], dtype=label_dtype)
        graph_list.append(g)
        labels.append(s['label'])
    batch = Batch.from_data_list(graph_list).to(device)
    return batch


def build_attfpgnn_batch(data_list: list, device: torch.device):
    """
    Collate AttFPGNN-adapted Data objects into a PyG Batch.

    AttFPGNN sử dụng `adapt_sample_to_attfpgnn` trả về torch_geometric.data.Data
    (không phải dict), với `.y` và `.smiles` đã được gắn sẵn.
    Hàm này dùng thay cho `build_fsgnntr_batch` trong AttFPGNN pipeline.

    Args:
        data_list: list of torch_geometric.data.Data, mỗi item có
                   .x (long), .edge_index, .edge_attr (long), .y (long), .smiles (str)
        device: torch.device

    Returns:
        batch: PyG Batch đã chuyển lên device
    """
    batch = Batch.from_data_list(data_list).to(device)
    return batch


# ─── Loss computation ──────────────────────────────────────────────────────────

def compute_gnn_loss(model, batch, criterion):
    """
    Forward pass + BCE loss for GNN_prediction model.
    Returns scalar loss.
    """
    graph_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    y = batch.y.view(graph_pred.shape).to(torch.float64)
    loss = criterion(graph_pred.double(), y)
    return loss.sum() / graph_pred.shape[0]


def compute_gnn_auroc(model, batch):
    """
    Evaluate GNN_prediction on batch, return ROC-AUC score.
    Returns float or NaN if only one class in labels.
    """
    with torch.no_grad():
        graph_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        proba = torch.sigmoid(graph_pred).squeeze(1).cpu().numpy()
        y_true = batch.y.view(-1).cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return float('nan')
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return float('nan')


# ─── MAML inner update ─────────────────────────────────────────────────────────

def maml_inner_update(model, support_batch, criterion, lr_update: float, n_steps: int = 1):
    """
    Perform n_steps of MAML inner gradient updates on support_batch.
    Modifies model parameters IN-PLACE temporarily.
    Returns the ORIGINAL parameter vector (for restoration).

    Usage:
        original = maml_inner_update(model, support_batch, criterion, lr_update)
        # ... evaluate on query ...
        vector_to_parameters(original, model.parameters())  # restore
    """
    original_params = parameters_to_vector(model.parameters()).detach().clone()

    for step in range(n_steps):
        retain = (step < n_steps - 1)
        loss = compute_gnn_loss(model, support_batch, criterion)

        grads = torch.autograd.grad(
            loss, model.parameters(),
            retain_graph=retain,
            allow_unused=True,
        )

        grad_list = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, model.parameters())
        ]
        grad_vector = parameters_to_vector(grad_list)
        param_vector = parameters_to_vector(model.parameters())
        updated = param_vector - lr_update * grad_vector
        vector_to_parameters(updated, model.parameters())

    return original_params  # caller must restore


# ─── Full meta-train step (one episode) ────────────────────────────────────────

def meta_train_step_gnn(
    model,
    optimizer,
    task_data: dict,
    K_shot: int,
    Q_query: int,
    device: torch.device,
    criterion,
    lr_update: float,
    n_inner_steps: int,
    create_meta_task_fn,
    adapt_sample_fn,
):
    """
    One FOMAML training episode for a GNN-only baseline.

    1. Sample support + query via create_meta_task_fn
    2. Adapt graphs to FS-GNNTR format via adapt_sample_fn
    3. MAML inner update on support (n_inner_steps)
    4. Compute query loss + query AUROC with adapted params
    5. Backward query loss → accumulate .grad
    6. Restore original params

    The caller is responsible for calling optimizer.zero_grad() before and
    optimizer.step() after (supports gradient accumulation over N episodes).

    Returns: (query_loss: Tensor, query_auroc: float)
    """
    try:
        support, query = create_meta_task_fn(task_data, K_shot, Q_query, train=True)
    except ValueError:
        return None, float('nan')

    if len(support) == 0 or len(query) == 0:
        return None, float('nan')

    support_adapted = [adapt_sample_fn(s) for s in support]
    query_adapted   = [adapt_sample_fn(q) for q in query]

    support_batch = build_fsgnntr_batch(support_adapted, device)
    query_batch   = build_fsgnntr_batch(query_adapted, device)

    model.train()
    original_params = maml_inner_update(
        model, support_batch, criterion, lr_update, n_steps=n_inner_steps
    )

    # Query loss with adapted params (meta-gradient)
    query_loss = compute_gnn_loss(model, query_batch, criterion)

    # Compute query AUROC for monitoring (no_grad)
    model.eval()
    query_auroc = compute_gnn_auroc(model, query_batch)
    model.train()

    query_loss.backward()  # accumulate gradients

    # Restore original params
    vector_to_parameters(original_params, model.parameters())

    return query_loss.detach(), query_auroc


# ─── Full meta-test step (one episode) ─────────────────────────────────────────

def meta_test_step_gnn(
    model,
    support_samples: list,
    query_samples: list,
    device: torch.device,
    criterion,
    lr_update: float,
    n_inner_steps: int,
    adapt_sample_fn,
):
    """
    One FOMAML test episode for a GNN-only baseline.

    1. Adapt graphs
    2. MAML inner adaptation on support (n_inner_steps, no meta-gradient needed)
    3. Evaluate on query → AUROC
    4. Restore params

    Returns: (auroc: float)
    """
    support_adapted = [adapt_sample_fn(s) for s in support_samples if s is not None]
    query_adapted   = [adapt_sample_fn(q) for q in query_samples  if q is not None]

    if len(support_adapted) == 0 or len(query_adapted) == 0:
        return float('nan')

    support_batch = build_fsgnntr_batch(support_adapted, device)
    query_batch   = build_fsgnntr_batch(query_adapted, device)

    # Save original params
    original_params = parameters_to_vector(model.parameters()).detach().clone()

    model.train()  # need train mode for MAML inner update (batch norm etc.)
    with torch.enable_grad():
        for step in range(n_inner_steps):
            retain = (step < n_inner_steps - 1)
            loss = compute_gnn_loss(model, support_batch, criterion)
            grads = torch.autograd.grad(
                loss, model.parameters(),
                retain_graph=retain, allow_unused=True
            )
            grad_list = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, model.parameters())
            ]
            updated = parameters_to_vector(model.parameters()) - lr_update * parameters_to_vector(grad_list)
            vector_to_parameters(updated, model.parameters())

    model.eval()
    auroc = compute_gnn_auroc(model, query_batch)

    # Restore
    vector_to_parameters(original_params, model.parameters())

    return auroc
