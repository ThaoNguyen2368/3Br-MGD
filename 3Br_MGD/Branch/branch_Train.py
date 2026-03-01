import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from data import load_all_splits
from branch_Build import build_model, VARIANT_NAMES, VARIANT_INFO
from BrMGD_train import train_meta_epoch, valid_meta_tasks

def train_branch(
    variant: str,
    meta_train: dict,
    meta_val: dict,
    device: torch.device,
    K_shot: int,
    Q_query: int,
    max_epochs: int,
    patience: int,
    train_episodes: int,
    val_episodes: int,
    lr: float,
) -> tuple:
    
    info = VARIANT_INFO[variant]
    print(f"  Variant: {info['name']}  |  branches: {info['branches']}")

    protonet  = build_model(variant, device)
    optimizer = torch.optim.Adam(protonet.parameters(), lr=lr)

    best_val_auroc   = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses     = []
    val_aurocs_hist  = []

    for epoch in range(max_epochs):

        # --- Train ---
        train_loss = train_meta_epoch(
            protonet, optimizer, meta_train,
            device, K_shot, Q_query, train_episodes,
        )
        train_losses.append(train_loss)

        # --- Validate ---
        val_auroc = float('nan')
        if meta_val:
            val_results = valid_meta_tasks(
                protonet, meta_val,
                device, K_shot, Q_query, val_episodes,
            )
            aurocs = [
                r['auroc'] for r in val_results.values()
                if not np.isnan(r['auroc'])
            ]
            val_auroc = float(np.mean(aurocs)) if aurocs else float('nan')

        val_aurocs_hist.append(val_auroc)

        if not np.isnan(val_auroc):
            if val_auroc > best_val_auroc:
                best_val_auroc   = val_auroc
                best_model_state = deepcopy(protonet.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            if epoch >= 20 and best_model_state is None:
                best_model_state = deepcopy(protonet.state_dict())

        # --- Log each 10 epoch ---
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: loss={train_loss:.4f}, "
                  f"val_auroc={val_auroc:.4f}")

        # --- Early stopping ---
        if (meta_val
                and patience_counter >= patience
                and best_model_state is not None):
            print(f"    Early stopping epoch={epoch}, "
                  f"best_val_auroc={best_val_auroc:.4f}")
            break

    if best_model_state is not None:
        protonet.load_state_dict(best_model_state)
        print(f"    Loaded best model (val_auroc={best_val_auroc:.4f})")

    return protonet, train_losses, val_aurocs_hist, best_val_auroc


def main():
    parser = argparse.ArgumentParser(
        description='Meta-training for branch variant'
    )
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default='checkpoints')
    parser.add_argument('--dataset',        type=str, default='tox21')
    parser.add_argument('--variants',       type=str, nargs='+', default=['all'],
                        help=f"List variant or 'all'. "
                             f": {VARIANT_NAMES}")
    parser.add_argument('--shots',          type=int, nargs='+', default=[5, 10])
    parser.add_argument('--max_epochs',     type=int, default=100)
    parser.add_argument('--patience',       type=int, default=15)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--val_episodes',   type=int, default=30)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--q_query',        type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.variants == ['all']:
        variants_to_run = VARIANT_NAMES
    else:
        variants_to_run = []
        for v in args.variants:
            if v not in VARIANT_NAMES:
                print(f"WARNING: variant '{v}' invalid, skip.")
            else:
                variants_to_run.append(v)

    if not variants_to_run:
        print("Invalid.")
        return

    print(f"Device   : {device}")
    print(f"Dataset  : {args.dataset}")
    print(f"Shots    : {args.shots}")
    print(f"Variants : {variants_to_run}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    meta_train, meta_val, meta_test = load_all_splits(args.data_dir)

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        patience  = args.patience + (5 if K_shot == 10 else 0)

        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")

        for variant in variants_to_run:
            info = VARIANT_INFO[variant]
            print(f"\n── {info['name']} ──")

            # Train
            protonet, train_losses, val_aurocs, best_val = train_branch(
                variant        = variant,
                meta_train     = meta_train,
                meta_val       = meta_val,
                device         = device,
                K_shot         = K_shot,
                Q_query        = args.q_query,
                max_epochs     = args.max_epochs,
                patience       = patience,
                train_episodes = args.train_episodes,
                val_episodes   = args.val_episodes,
                lr             = args.lr,
            )

            ckpt_name = f"{variant}_{args.dataset}_{shot_name}_best.pt"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            torch.save({
                'model_state': protonet.state_dict(),
                'variant':     variant,
                'val_auroc':   best_val,
                'epoch':       len(train_losses),
                'config': {
                    'dataset':         args.dataset,
                    'variant':         variant,
                    'K_shot':          K_shot,
                    'Q_query':         args.q_query,
                    'train_episodes':  args.train_episodes,
                    'val_episodes':    args.val_episodes,
                    'lr':              args.lr,
                    'patience':        patience,
                },
            }, ckpt_path)
            print(f"    Checkpoint → {ckpt_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()