import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy

from data import load_all_splits
from BrMGD_model import TripleEncoder, EnhancedProtoNet
from BrMGD_eval import collate_batch, evaluate_meta_task


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
    criterion       = nn.CrossEntropyLoss()
    total_loss      = 0.0
    n_success       = 0

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
        query_y = torch.tensor(
            [class_to_idx[y.item()] for y in qry_y_raw],
            dtype=torch.long, device=device,
        )

        loss = criterion(logits, query_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_success  += 1

    return total_loss / n_success if n_success > 0 else 0.0


def valid_meta_tasks(
    protonet, datasets_dict: dict,
    device, K_shot: int, Q_query: int, episodes: int,
) -> dict:
    results = {}

    for task_name, task_data in datasets_dict.items():
        accs, f1s, aurocs = [], [], []

        for _ in range(episodes):
            try:
                support, query = create_meta_task(task_data, K_shot, Q_query, train=True)
                acc, f1, auroc = evaluate_meta_task(protonet, support, query, device)
                if not np.isnan(acc):   accs.append(acc)
                if not np.isnan(f1):    f1s.append(f1)
                if not np.isnan(auroc): aurocs.append(auroc)
            except ValueError:
                continue

        results[task_name] = {
            'acc':       np.mean(accs)   if accs   else float('nan'),
            'f1':        np.mean(f1s)    if f1s    else float('nan'),
            'auroc':     np.mean(aurocs) if aurocs else float('nan'),
            'acc_std':   np.std(accs)    if accs   else 0.0,
            'f1_std':    np.std(f1s)     if f1s    else 0.0,
            'auroc_std': np.std(aurocs)  if aurocs else 0.0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Meta-training 3BRMGD')
    parser.add_argument('--data_dir',        type=str,   required=True)
    parser.add_argument('--output_dir',      type=str,   default='checkpoints')
    parser.add_argument('--dataset',         type=str,   default='tox21',
                        choices=['tox21', 'sider'])
    parser.add_argument('--shots',           type=int,   nargs='+', default=[5, 10])
    parser.add_argument('--max_epochs',      type=int,   default=100)
    parser.add_argument('--patience',        type=int,   default=15)
    parser.add_argument('--train_episodes',  type=int,   default=100)
    parser.add_argument('--val_episodes',    type=int,   default=30)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--q_query',         type=int,   default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {args.shots}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    meta_train, meta_val, meta_test = load_all_splits(args.data_dir)

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        patience  = args.patience + (5 if K_shot == 10 else 0)

        print(f"\n{'='*25} {shot_name.upper()} {'='*25}")
        print(f"K_shot={K_shot}, Q_query={args.q_query}, "
              f"patience={patience}, train_ep={args.train_episodes}")

        encoder  = TripleEncoder().to(device)
        protonet = EnhancedProtoNet(encoder).to(device)
        optimizer = torch.optim.Adam(protonet.parameters(), lr=args.lr)

        best_val_auroc   = 0.0
        best_model_state = None
        patience_counter = 0
        train_losses     = []
        val_aurocs       = []

        for epoch in range(args.max_epochs):

            # --- Train ---
            train_loss = train_meta_epoch(
                protonet, optimizer, meta_train,
                device, K_shot, args.q_query, args.train_episodes,
            )
            train_losses.append(train_loss)

            # --- Validate ---
            if meta_val:
                val_results = valid_meta_tasks(
                    protonet, meta_val,
                    device, K_shot, args.q_query, args.val_episodes,
                )
                val_auroc = float(np.nanmean([
                    r['auroc'] for r in val_results.values()
                ]))
                val_aurocs.append(val_auroc)

                if val_auroc > best_val_auroc:
                    best_val_auroc   = val_auroc
                    best_model_state = deepcopy(protonet.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                val_aurocs.append(float('nan'))
                if epoch >= 20:
                    best_model_state = deepcopy(protonet.state_dict())

            # --- Log ---
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, "
                      f"val_auroc={val_aurocs[-1]:.4f}")

            # --- Early stopping ---
            if meta_val and patience_counter >= patience and best_model_state is not None:
                print(f"  Early stopping at epoch {epoch}, "
                      f"best val AUROC = {best_val_auroc:.4f}")
                break

        # --- Load best model ---
        if best_model_state is not None:
            protonet.load_state_dict(best_model_state)
            print(f"  Loaded best model (val AUROC = {best_val_auroc:.4f})")

        # --- Save checkpoint ---
        ckpt_path = os.path.join(
            args.output_dir,
            f"BrMGD_{args.dataset}_{shot_name}_best.pt"
        )
        torch.save({
            'model_state': protonet.state_dict(),
            'val_auroc':   best_val_auroc,
            'epoch':       len(train_losses),
            'config': {
                'dataset':         args.dataset,
                'K_shot':          K_shot,
                'Q_query':         args.q_query,
                'train_episodes':  args.train_episodes,
                'val_episodes':    args.val_episodes,
                'lr':              args.lr,
                'patience':        patience,
            },
        }, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")


    print("\nTraining complete!")


if __name__ == '__main__':
    main()