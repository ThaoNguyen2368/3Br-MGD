import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import roc_auc_score
import sys
# Thêm thư mục chứa script này vào path để import local
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_all_splits
from BrMGD_model import TripleEncoder, EnhancedProtoNet
from BrMGD_eval import collate_batch
import torch.nn.functional as F


def create_meta_task(task_data: dict, K_shot: int, Q_query: int, train: bool = True):
    pos_pool = task_data['pos']
    neg_pool = task_data['neg']

    if len(pos_pool) < K_shot or len(neg_pool) < K_shot:
        raise ValueError(
            f"Not enough sample: pos={len(pos_pool)}, neg={len(neg_pool)}"
        )

    # --- Support set ---
    sup_pos = random.sample(pos_pool, K_shot)
    sup_neg = random.sample(neg_pool, K_shot)
    support = sup_pos + sup_neg
    random.shuffle(support)

    # --- Query set ---
    sup_pos_set = set(id(s) for s in sup_pos)
    sup_neg_set = set(id(s) for s in sup_neg)

    remaining_pos = [s for s in pos_pool if id(s) not in sup_pos_set]
    remaining_neg = [s for s in neg_pool if id(s) not in sup_neg_set]

    if train:
        q_pos = random.sample(remaining_pos, min(Q_query, len(remaining_pos)))
        q_neg = random.sample(remaining_neg, min(Q_query, len(remaining_neg)))
    else:
        q_pos = remaining_pos
        q_neg = remaining_neg

    query = q_pos + q_neg
    random.shuffle(query)

    return support, query


def train_meta_epoch(
    protonet, optimizer, datasets_dict: dict,
    device, K_shot: int, Q_query: int, episodes: int,
) -> float:
    protonet.train()
    total_loss = 0.0
    all_query_aurocs = []
    successful_episodes = 0
    criterion = nn.CrossEntropyLoss()

    task_names = list(datasets_dict.keys())

    for _ in range(episodes):
        task_name = random.choice(task_names)
        task_data = datasets_dict[task_name]

        try:
            support, query = create_meta_task(task_data, K_shot, Q_query, train=True)
        except ValueError:
            continue

        sup_fp, sup_graph, sup_seq, support_y = collate_batch(support, device)
        qry_fp, qry_graph, qry_seq, qry_y_raw = collate_batch(query,   device)

        optimizer.zero_grad()
        logits, class_to_idx = protonet(
            sup_fp, sup_graph, sup_seq, support_y,
            qry_fp, qry_graph, qry_seq,
        )
        probs = F.softmax(logits, dim=1)
        if 1 in class_to_idx:
            pos_idx = class_to_idx[1]
            y_true = (qry_y_raw == 1).cpu().numpy()
            y_scores = probs[:, pos_idx].detach().cpu().numpy()
            try:
                all_query_aurocs.append(roc_auc_score(y_true, y_scores))
            except: pass
        # -------------------------

        query_y = torch.tensor([class_to_idx[y.item()] for y in qry_y_raw], dtype=torch.long, device=device)
        loss = criterion(logits, query_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        successful_episodes += 1

    return (total_loss / successful_episodes), np.mean(all_query_aurocs)

def main():
    parser = argparse.ArgumentParser(description='Meta-training 3BRMGD')
    parser.add_argument('--data_dir',        type=str,   required=True)
    parser.add_argument('--output_dir',      type=str,   default='checkpoints')
    parser.add_argument('--dataset',         type=str,   default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',           type=int,   nargs='+', default=[5, 10])
    parser.add_argument('--max_epochs',      type=int,   default=200)
    parser.add_argument('--patience',        type=int,   default=20)
    parser.add_argument('--train_episodes',  type=int,   default=100)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--q_query',         type=int,   default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {args.shots}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    meta_train, _ = load_all_splits(args.data_dir)

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        patience  = args.patience 

        print(f"\n{'='*25} {shot_name.upper()} {'='*25}")
        print(f"K_shot={K_shot}, Q_query={args.q_query}, "
              f"patience={patience}, train_ep={args.train_episodes}")

        encoder  = TripleEncoder().to(device)
        protonet = EnhancedProtoNet(encoder).to(device)
        optimizer = torch.optim.Adam(protonet.parameters(), lr=args.lr)

        best_query_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        train_losses     = []

        for epoch in range(1, args.max_epochs + 1):

            # --- Train ---

            train_loss, train_query_auroc = train_meta_epoch(
                protonet, optimizer, meta_train,
                device, K_shot, args.q_query, args.train_episodes,
            )
            train_losses.append(train_loss)

            if train_query_auroc > (best_query_auroc + 0.001):
                best_query_auroc = train_query_auroc
                best_model_state = deepcopy(protonet.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # --- Log ---
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, "
                      f"train_query_auc={train_query_auroc:.4f}, patience={patience_counter}/{patience}")

            # --- Early stopping ---
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}. Best Query AUC = {best_query_auroc:.4f}")
                break

        # --- Save Best Checkpoint ---
        if best_model_state is not None:
            ckpt_path = os.path.join(args.output_dir, f"BrMGD_{args.dataset}_{shot_name}_best.pth")
            torch.save({
                'model_state': best_model_state,
                'val_auroc':   best_query_auroc, 
                'epoch':       epoch,
                'config':      vars(args),
            }, ckpt_path)
            print(f"   Checkpoint saved → {ckpt_path}")

    print("\nTraining complete!")

if __name__ == '__main__':
    main()