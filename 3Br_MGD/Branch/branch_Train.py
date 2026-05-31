import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
# Thêm đường dẫn tới thư mục Br_MGD để import data và các hàm bổ trợ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Br_MGD")))

from data import load_all_splits
from branch_Build import build_model, VARIANT_NAMES, VARIANT_INFO
from BrMGD_train import train_meta_epoch


def set_seed(seed: int = 42):
    """Fix tất cả nguồn ngẫu nhiên để đảm bảo reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_branch(
    variant: str,
    meta_train: dict,
    device: torch.device,
    K_shot: int,
    Q_query: int,
    max_epochs: int,
    patience: int,
    train_episodes: int,
    lr: float,
) -> tuple:
    
    info = VARIANT_INFO[variant]
    print(f"   Variant: {info['name']}  |  branches: {info['branches']}")

    protonet  = build_model(variant, device)
    optimizer = torch.optim.Adam(protonet.parameters(), lr=lr)

    best_query_auroc = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses     = []
    query_aurocs_hist = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_query_auroc = train_meta_epoch(
            protonet, optimizer, meta_train,
            device, K_shot, Q_query, train_episodes
        )
        train_losses.append(train_loss)
        query_aurocs_hist.append(train_query_auroc)

        if train_query_auroc > (best_query_auroc + 0.001):
            best_query_auroc = train_query_auroc
            best_model_state = deepcopy(protonet.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}: loss={train_loss:.4f}, "
                  f"train_query_auc={train_query_auroc:.4f}, patience={patience_counter}/{patience}")

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f"      Early stopping at epoch {epoch}. Best Query AUC = {best_query_auroc:.4f}")
            break

    if best_model_state is not None:
        protonet.load_state_dict(best_model_state)

    return protonet, train_losses, query_aurocs_hist, best_query_auroc


def main():
    parser = argparse.ArgumentParser(description='Meta-training for branch variant')
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default='checkpoints')
    parser.add_argument('--dataset',        type=str, default='tox21')
    parser.add_argument('--variants',       type=str, nargs='+', default=['all'],
                        help=f"List variant or 'all'. : {VARIANT_NAMES}")
    parser.add_argument('--shots',           type=int, nargs='+', default=[5, 10])
    parser.add_argument('--max_epochs',      type=int, default=100)
    parser.add_argument('--patience',        type=int, default=20)
    parser.add_argument('--train_episodes',  type=int, default=100)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--q_query',         type=int, default=128) # Tăng lên 128 theo protocol
    parser.add_argument('--seed',            type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.variants == ['all']:
        variants_to_run = VARIANT_NAMES
    else:
        variants_to_run = [v for v in args.variants if v in VARIANT_NAMES]

    if not variants_to_run:
        print("No valid variants to run.")
        return

    print(f"Device   : {device}")
    print(f"Dataset  : {args.dataset}")
    print(f"Shots    : {args.shots}")
    print(f"Variants : {variants_to_run}")
    print(f"Seed     : {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    meta_train, _ = load_all_splits(args.data_dir)

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        current_patience = args.patience + (5 if K_shot == 10 else 0)

        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")

        for variant in variants_to_run:
            info = VARIANT_INFO[variant]
            print(f"\n── {info['name']} ──")

            # Gọi hàm train
            protonet, train_losses, query_aurocs, best_val = train_branch(
                variant        = variant,
                meta_train     = meta_train,
                device         = device,
                K_shot         = K_shot,
                Q_query        = args.q_query,
                max_epochs     = args.max_epochs,
                patience       = current_patience,
                train_episodes = args.train_episodes,
                lr             = args.lr,
            )

            # Lưu checkpoint tại hàm main để có quyền truy cập vào args và shot_name
            ckpt_name = f"{variant}_{args.dataset}_{shot_name}_best.pth" # Dùng đuôi .pth cho đồng bộ
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            
            torch.save({
                'model_state': protonet.state_dict(),
                'variant':     variant,
                'val_auroc':   best_val, # Chính là best query auroc
                'epoch':       len(train_losses),
                'config':      vars(args),
            }, ckpt_path)
            print(f"       Checkpoint saved → {ckpt_path}")

    print("\nTraining complete!")

if __name__ == '__main__':
    main()