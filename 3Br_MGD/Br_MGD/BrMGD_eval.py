import torch
import numpy as np
from torch_geometric.data import Batch
from sklearn.metrics import f1_score, roc_auc_score


def collate_batch(samples: list, device: torch.device):
    """
    Stack list of sample dicts thành batch tensors.

    Args:
        samples : list of {'fp', 'graph', 'sequence', 'label', ...}
        device  : torch.device

    Returns:
        fp_tensor    : [B, 2048]
        graph_batch  : torch_geometric.data.Batch
        seq_tensor   : [B, 200]
        label_tensor : [B]  dtype=torch.long
    """
    fp_tensor    = torch.stack([s['fp']       for s in samples]).to(device)
    seq_tensor   = torch.stack([s['sequence'] for s in samples]).to(device)
    graph_batch  = Batch.from_data_list([s['graph'] for s in samples]).to(device)
    label_tensor = torch.tensor([s['label']   for s in samples],
                                dtype=torch.long, device=device)
    return fp_tensor, graph_batch, seq_tensor, label_tensor


def evaluate_meta_task(protonet, support: list, query: list, device: torch.device):
    protonet.eval()
    with torch.no_grad():

        # --- Support set ---
        sup_fp, sup_graph, sup_seq, support_y = collate_batch(support, device)

        # --- Query set ---
        qry_fp, qry_graph, qry_seq, query_y_raw = collate_batch(query, device)

        # --- Forward ---
        logits, class_to_idx = protonet(
            sup_fp, sup_graph, sup_seq, support_y,
            qry_fp, qry_graph, qry_seq,
        )

        # Remap query labels to prototype index space
        query_y = torch.tensor(
            [class_to_idx[y.item()] for y in query_y_raw],
            dtype=torch.long,
            device=device,
        )

        # --- Accuracy ---
        preds = torch.argmax(logits, dim=1)
        acc   = (preds == query_y).float().mean().item()

        # --- F1 ---
        if len(torch.unique(query_y)) < 2:
            f1 = float('nan')
        else:
            try:
                f1 = f1_score(
                    query_y.cpu().numpy(),
                    preds.cpu().numpy(),
                    average='binary',
                    zero_division=0,
                )
            except Exception:
                f1 = float('nan')

        # --- AUROC ---
        pos_index = class_to_idx.get(1, None)
        if pos_index is None or len(torch.unique(query_y)) < 2:
            auroc = float('nan')
        else:
            try:
                probs = torch.softmax(logits, dim=1)[:, pos_index]
                auroc = roc_auc_score(
                    query_y.cpu().numpy(),
                    probs.detach().cpu().numpy(),
                )
            except Exception:
                auroc = float('nan')

    return acc, f1, auroc