import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModel
import sys
# Thêm đường dẫn tới thư mục Br_MGD để import 'data'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Br_MGD")))

from data import load_all_splits


def set_seed(seed: int = 42):
    """Fix tất cả nguồn ngẫu nhiên để đảm bảo reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ChemBERTaProtoNet(nn.Module):
    def __init__(
        self,
        checkpoint: str = "DeepChemChemBERTa-77M-MLM",
        proj_dim: int = 128,
        max_length: int = 128,
        freeze_bert: bool = True,
    ):
        super().__init__()
        print(f"Loading tokenizer {checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"Loading BERT model {checkpoint}")
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.max_length = max_length

        hidden_size = self.bert.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.projection.parameters())
            print(f"BERT frozen. Trainable params {trainable} (projection only)")
        else:
            total = sum(p.numel() for p in self.parameters())
            print(f"Full fine-tune. Total params {total}")

        self.freeze_bert = freeze_bert

    def encode(self, smiles_list: list, device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        if self.freeze_bert:
            with torch.no_grad():
                output = self.bert(**encoded)
        else:
            output = self.bert(**encoded)

        cls_emb = output.last_hidden_state[:, 0, :]
        return self.projection(cls_emb)

    def forward(
        self,
        support_smiles: list,
        support_y: torch.Tensor,
        query_smiles: list,
        device: torch.device,
    ):
        support_emb = self.encode(support_smiles, device)
        query_emb = self.encode(query_smiles, device)

        classes = torch.unique(support_y)
        class_to_idx = {c.item(): i for i, c in enumerate(classes)}
        prototypes = torch.stack([
            support_emb[support_y == c].mean(0) for c in classes
        ])

        n, m = query_emb.size(0), prototypes.size(0)
        dists = (
            query_emb.unsqueeze(1).expand(n, m, -1) -
            prototypes.unsqueeze(0).expand(n, m, -1)
        ).pow(2).sum(dim=2)

        logits = -dists
        return logits, class_to_idx


def create_episode(task_data: dict, K_shot: int, Q_query: int, train: bool = True):
    pos_pool = task_data['pos']
    neg_pool = task_data['neg']

    if len(pos_pool) < K_shot + 1 or len(neg_pool) < K_shot + 1:
        raise ValueError(f"Not enough pos={len(pos_pool)}, neg={len(neg_pool)}")

    sup_pos = random.sample(pos_pool, K_shot)
    sup_neg = random.sample(neg_pool, K_shot)
    support = sup_pos + sup_neg
    random.shuffle(support)

    sup_pos_ids = {id(s) for s in sup_pos}
    sup_neg_ids = {id(s) for s in sup_neg}
    remaining_pos = [s for s in pos_pool if id(s) not in sup_pos_ids]
    remaining_neg = [s for s in neg_pool if id(s) not in sup_neg_ids]

    if train:
        q_pos = random.sample(remaining_pos, min(Q_query, len(remaining_pos)))
        q_neg = random.sample(remaining_neg, min(Q_query, len(remaining_neg)))
    else:
        q_pos = remaining_pos
        q_neg = remaining_neg

    query = q_pos + q_neg
    random.shuffle(query)
    return support, query


def episode_to_smiles(support: list, query: list, device: torch.device):
    sup_smiles = [s['smiles'] for s in support]
    support_y = torch.tensor([s['label'] for s in support],
                             dtype=torch.long, device=device)
    qry_smiles = [s['smiles'] for s in query]
    query_y = torch.tensor([s['label'] for s in query],
                           dtype=torch.long, device=device)
    return sup_smiles, support_y, qry_smiles, query_y


def evaluate_episode(model, support, query, device):
    model.eval()
    with torch.no_grad():
        sup_smiles, support_y, qry_smiles, query_y_raw = episode_to_smiles(support, query, device)

        unique_q = set(query_y_raw.cpu().numpy().tolist())
        unique_s = set(support_y.cpu().numpy().tolist())
        if len(unique_q) < 2 or len(unique_s) < 2:
            return float('nan'), float('nan'), float('nan'), float('nan')
        if not unique_q.issubset(unique_s):
            return float('nan'), float('nan'), float('nan'), float('nan')

        logits, class_to_idx = model(sup_smiles, support_y, qry_smiles, device)
        query_y_remapped = torch.tensor(
            [class_to_idx[y.item()] for y in query_y_raw],
            dtype=torch.long, device=device,
        )
        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_y_remapped).float().mean().item()

        pos_idx = class_to_idx.get(1, None)
        if pos_idx is None:
            return acc, float('nan'), float('nan'), float('nan')

        try:
            y_true_b = (query_y_remapped == pos_idx).cpu().numpy().astype(int)
            y_pred_b = (preds == pos_idx).cpu().numpy().astype(int)
            f1 = f1_score(y_true_b, y_pred_b, average='binary', zero_division=0)
        except Exception:
            f1 = float('nan')

        try:
            probs = torch.softmax(logits, dim=1)[:, pos_idx]
            auroc = roc_auc_score(
                query_y_raw.cpu().numpy(),
                probs.detach().cpu().numpy(),
            )
            auprc = average_precision_score(
                query_y_raw.cpu().numpy(),
                probs.detach().cpu().numpy(),
            )
        except Exception:
            auroc = float('nan')
            auprc = float('nan')

    return acc, f1, auroc, auprc


def train_chemberta(
    model, meta_train, meta_val, device,
    K_shot, Q_query, max_epochs, patience,
    train_episodes, val_episodes, lr,
):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auroc = 0.0
    best_state = None
    patience_counter = 0
    train_losses = []
    val_aurocs_hist = []
    task_names = list(meta_train.keys())

    for epoch in range(max_epochs):
        model.train()
        total_loss, n_ok = 0.0, 0
        for _ in range(train_episodes):
            task_name = random.choice(task_names)
            try:
                support, query = create_episode(meta_train[task_name], K_shot, Q_query, train=True)
                sup_smiles, support_y, qry_smiles, query_y_raw = episode_to_smiles(support, query, device)

                if len(torch.unique(support_y)) < 2:
                    continue

                optimizer.zero_grad()
                logits, class_to_idx = model(sup_smiles, support_y, qry_smiles, device)
                query_y = torch.tensor(
                    [class_to_idx[y.item()] for y in query_y_raw],
                    dtype=torch.long, device=device,
                )
                loss = criterion(logits, query_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_ok += 1
            except Exception:
                continue

        train_loss = total_loss / n_ok if n_ok > 0 else 0.0
        train_losses.append(train_loss)

        val_auroc = float('nan')
        if meta_val:
            aurocs = []
            for task_data in meta_val.values():
                for _ in range(val_episodes):
                    try:
                        support, query = create_episode(task_data, K_shot, Q_query, train=True)
                        _, _, a, _ = evaluate_episode(model, support, query, device)
                        if not np.isnan(a):
                            aurocs.append(a)
                    except Exception:
                        continue
            val_auroc = float(np.mean(aurocs)) if aurocs else float('nan')

        val_aurocs_hist.append(val_auroc)

        if not np.isnan(val_auroc):
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            if epoch == 10 and best_state is None:
                best_state = deepcopy(model.state_dict())

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} loss={train_loss:.4f}, val_auroc={val_auroc:.4f}")

        if meta_val and patience_counter >= patience and best_state is not None:
            print(f"Early stopping epoch={epoch}, best={best_val_auroc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_aurocs_hist, best_val_auroc

def evaluate_and_write(
    model, meta_test, device,
    K_shot, Q_query, test_episodes,
    dataset, shot_name, result_dir,
    checkpoint_name,
):
    os.makedirs(result_dir, exist_ok=True)
    WRITE_LIMIT = 30

    task_names = list(meta_test.keys())
    n_tasks = len(task_names)
    exp = [[] for _ in range(n_tasks)]
    auprc_exp = [[] for _ in range(n_tasks)]
    acc_exp = [[] for _ in range(n_tasks)]
    f1_exp = [[] for _ in range(n_tasks)]

    shot_num = shot_name.replace('-shot', '')
    result_file = os.path.join(result_dir, f"mean-ChemBERTa_{dataset}_{shot_num}shot.txt")

    with open(result_file, 'a', encoding='utf-8') as rf:
        rf.write(f"{'='*65}\n")
        rf.write(f"Model    ChemBERTa ({checkpoint_name})\n")
        rf.write(f"Dataset  {dataset}    {shot_name}    K_shot={K_shot}\n")
        rf.write(f"Tasks    {task_names}\n")
        rf.write(f"{'='*65}\n")

    print(f"\nEVALUATING ChemBERTa ({test_episodes} episodes)")

    for ep in range(1, test_episodes + 1):
        for i, task_name in enumerate(task_names):
            task_data = meta_test[task_name]
            try:
                support, query = create_episode(task_data, K_shot, Q_query, train=True)
                acc, f1, auroc, auprc = evaluate_episode(model, support, query, device)
                exp[i].append(round(auroc, 4) if not np.isnan(auroc) else 0.0)
                auprc_exp[i].append(round(auprc, 4) if not np.isnan(auprc) else 0.0)
                if not np.isnan(acc):
                    acc_exp[i].append(acc)
                if not np.isnan(f1):
                    f1_exp[i].append(f1)
            except Exception:
                exp[i].append(0.0)
                auprc_exp[i].append(0.0)

        if ep == WRITE_LIMIT:
            with open(result_file, 'a', encoding='utf-8') as rf:
                rf.write(f"Results AUROC\t{exp}\n")
                rf.write(f"Results AUPRC\t{auprc_exp}\n")

        if ep % 10 == 0:
            means = [f"{np.mean(exp[i]):.4f}" if exp[i] else 'nan' for i in range(n_tasks)]
            print(f"[ChemBERTa] Ep {ep:3d}/{test_episodes} — AUROC {means}")

    results = {}
    summary_lines = [
        f"{'-'*65}",
        f"{'Task':40} {'Acc':10} {'F1':10} {'AUROC':10} {'AUPRC':10}  (mean±std)",
        f"{'-'*65}",
    ]

    for i, task_name in enumerate(task_names):
        auroc_mean = np.mean(exp[i]) if exp[i] else float('nan')
        auroc_std = np.std(exp[i]) if len(exp[i]) > 1 else 0.0
        auprc_mean = np.mean(auprc_exp[i]) if auprc_exp[i] else float('nan')
        auprc_std = np.std(auprc_exp[i]) if len(auprc_exp[i]) > 1 else 0.0
        acc_mean = np.mean(acc_exp[i]) if acc_exp[i] else float('nan')
        acc_std = np.std(acc_exp[i]) if len(acc_exp[i]) > 1 else 0.0
        f1_mean = np.mean(f1_exp[i]) if f1_exp[i] else float('nan')
        f1_std = np.std(f1_exp[i]) if len(f1_exp[i]) > 1 else 0.0

        line = (f"{task_name:40} "
                f"{acc_mean:.4f}±{acc_std:.4f}  "
                f"{f1_mean:.4f}±{f1_std:.4f}  "
                f"{auroc_mean:.4f}±{auroc_std:.4f}  "
                f"{auprc_mean:.4f}±{auprc_std:.4f}")
        summary_lines.append(line)
        print(line)

        results[task_name] = {
            'auroc': auroc_mean, 'auroc_std': auroc_std,
            'auprc': auprc_mean, 'auprc_std': auprc_std,
            'acc': acc_mean, 'acc_std': acc_std,
            'f1': f1_mean, 'f1_std': f1_std,
        }

    all_auroc = [results[t]['auroc'] for t in task_names if not np.isnan(results[t]['auroc'])]
    all_auprc = [results[t]['auprc'] for t in task_names if not np.isnan(results[t]['auprc'])]
    all_acc = [results[t]['acc'] for t in task_names if not np.isnan(results[t]['acc'])]
    all_f1 = [results[t]['f1'] for t in task_names if not np.isnan(results[t]['f1'])]

    ov_auroc = np.mean(all_auroc) if all_auroc else float('nan')
    ov_auprc = np.mean(all_auprc) if all_auprc else float('nan')
    ov_acc = np.mean(all_acc) if all_acc else float('nan')
    ov_f1 = np.mean(all_f1) if all_f1 else float('nan')

    ov_line = (f"{'OVERALL AVERAGE':40} "
               f"{ov_acc:.4f}{'':12}"
               f"{ov_f1:.4f}{'':12}"
               f"{ov_auroc:.4f}{'':12}"
               f"{ov_auprc:.4f}")
    summary_lines.append(ov_line)
    summary_lines.append('-'*65)
    print(ov_line)

    with open(result_file, 'a', encoding='utf-8') as rf:
        rf.write('\n'.join(summary_lines) + '\n')
    print(f"Done! → {result_file}")

    return results, ov_auroc, ov_auprc


def main():
    parser = argparse.ArgumentParser(description='Evaluate ChemBERTa pretrained on meta-test tasks')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='tox21', choices=['tox21', 'sider'])
    parser.add_argument('--shots', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--checkpoint', type=str,
                        default='seyonecChemBERTa-zinc-base-v1',
                        help='HuggingFace checkpoint')
    parser.add_argument('--freeze_bert', action='store_true', default=True)
    parser.add_argument('--no_freeze', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--val_episodes', type=int, default=30)
    parser.add_argument('--test_episodes', type=int, default=30)
    parser.add_argument('--q_query', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of meta_train tasks dùng làm validation (default: 0.2)')
    args = parser.parse_args()

    # --- Fix seed TRƯỚC KHI split để split cũng deterministic ---
    set_seed(args.seed)

    freeze_bert = not args.no_freeze

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      {device}")
    print(f"Dataset     {args.dataset}")
    print(f"Shots       {args.shots}")
    print(f"Checkpoint  {args.checkpoint}")
    print(f"freeze_bert {freeze_bert}")
    print(f"Seed        {args.seed}")

    os.makedirs('results', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    meta_train_all, meta_test = load_all_splits(args.data_dir)

    # --- Tách meta_val ngẫu nhiên theo seed ---
    all_task_names = list(meta_train_all.keys())
    random.shuffle(all_task_names)          # seed đã được fix ở trên
    n_val = max(1, int(len(all_task_names) * args.val_ratio))
    val_keys   = all_task_names[:n_val]
    train_keys = all_task_names[n_val:]

    meta_val       = {k: meta_train_all[k] for k in val_keys}
    meta_train     = {k: meta_train_all[k] for k in train_keys}

    print(f"Val ratio   {args.val_ratio}  →  "
          f"train={len(meta_train)} tasks, val={len(meta_val)} tasks")
    print(f"Val tasks   {val_keys}")

    all_results = {}

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        patience = args.patience + (5 if K_shot == 10 else 0)

        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")

        model = ChemBERTaProtoNet(
            checkpoint=args.checkpoint,
            proj_dim=128,
            max_length=128,
            freeze_bert=freeze_bert,
        ).to(device)

        model, train_losses, val_aurocs, best_val = train_chemberta(
            model=model,
            meta_train=meta_train,
            meta_val=meta_val,
            device=device,
            K_shot=K_shot,
            Q_query=args.q_query,
            max_epochs=args.max_epochs,
            patience=patience,
            train_episodes=args.train_episodes,
            val_episodes=args.val_episodes,
            lr=args.lr,
        )
        print(f"Best val AUROC = {best_val:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"ChemBERTa_{args.dataset}_{shot_name}_best.pt")
        torch.save({
            'model_state': model.state_dict(),
            'checkpoint': args.checkpoint,
            'val_auroc': best_val,
            'K_shot': K_shot,
            'freeze_bert': freeze_bert,
        }, ckpt_path)
        print(f"Checkpoint → {ckpt_path}")

        task_results, overall_auroc, overall_auprc = evaluate_and_write(
            model=model,
            meta_test=meta_test,
            device=device,
            K_shot=K_shot,
            Q_query=args.q_query,
            test_episodes=args.test_episodes,
            dataset=args.dataset,
            shot_name=shot_name,
            result_dir='results',
            checkpoint_name=args.checkpoint,
        )

        all_results[shot_name] = {
            'overall_auroc': overall_auroc,
            'overall_auprc': overall_auprc,
            'per_task': task_results,
        }

    json_path = os.path.join('results', f'chemberta_{args.dataset}_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {s: {k: float(v) if isinstance(v, float) else v for k, v in r.items()}
             for s, r in all_results.items()},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Summary JSON → {json_path}")

    print(f"\n{'='*50}")
    print(f"ChemBERTa ({args.checkpoint}) — {args.dataset.upper()}")
    print(f"{'='*50}")
    print(f"{'Shot':12} {'Overall AUROC':15} {'Overall AUPRC':15}")
    print('-'*45)
    for shot_name, r in all_results.items():
        print(f"{shot_name:12} {r['overall_auroc']:<15.4f} {r['overall_auprc']:<15.4f}")
    print("\nDone!")


if __name__ == '__main__':
    main()