"""
test.py — Meta-testing entry point for AttFPGNN baseline.

Usage:
  python baselines/attfpgnn/test.py \
      --data_dir 3Br_MGD/Data/tox21/processed \
      --dataset tox21 \
      --shots 5 10 \
      --episodes_file baselines/episodes_seed42_tox21.json
"""

import sys, os, argparse, json
import torch
import numpy as np
from collections import defaultdict

# Path setup
_BASELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_BASELINE_ROOT, '..'))
_BRMGD_PATH    = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')
_ATTFPGNN_LOCAL = os.path.join(_BASELINE_ROOT, 'attfpgnn')

for p in [_BRMGD_PATH, _ATTFPGNN_LOCAL, _BASELINE_ROOT, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import baselines.attfpgnn.config as CFG
from baselines.seed_utils import set_seed
from baselines.attfpgnn.adapter import adapt_sample_to_attfpgnn
from baselines.attfpgnn.prepare_fps import generate_fingerprints_for_dataset

from data import load_all_splits
from episode_manager import load_episodes, reconstruct_sample_from_smiles
from maml_mol_relation_model import MamlMolRelationModel
from maml_mol_relation_trainer import MamlMolRelationTrainer
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from maml_utils import build_attfpgnn_batch

class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_test(CFG, args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Model   : {CFG.MODEL_NAME}")
    print(f"Dataset : {args.dataset}")

    os.makedirs(args.output_dir, exist_ok=True)

    fps_path = os.path.join(CFG.ATTFPGNN_DATA_DIR, "all_fps.npy")
    if not os.path.exists(fps_path):
        print("Fingerprints not found. Generating...")
        generate_fingerprints_for_dataset(args.dataset)
        import maml_mol_relation_model
        import importlib
        importlib.reload(maml_mol_relation_model)

    episodes = load_episodes(args.episodes_file)
    print(f"Loaded episodes from: {args.episodes_file}")

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

        # Build dummy args
        trainer_args = DummyArgs(
            device=device,
            dataset=args.dataset,
            test_dataset=args.dataset,
            data_dir=CFG.ATTFPGNN_DATA_DIR,
            train_tasks=[],
            test_tasks=[],
            n_shot_train=K_shot,
            n_shot_test=K_shot,
            n_query=args.q_query,
            emb_dim=CFG.EMB_DIM,
            batch_task=CFG.BATCH_TASK,
            update_step=1,
            update_step_test=CFG.UPDATE_STEP_TEST,
            inner_update_step=CFG.INNER_UPDATE_STEP,
            trial_path=args.output_dir,
            preload_train_data=False,
            preload_test_data=False,
            support_valid=0,
            save_logs=0,
            meta_lr=CFG.META_LR,
            inner_lr=CFG.INNER_LR,
            weight_decay=CFG.WEIGHT_DECAY,
            clip_value=1.0,
            enc_layer=CFG.GRAPH_LAYERS,
            JK=CFG.JK,
            dropout=CFG.DROPOUT,
            enc_pooling=CFG.POOLING,
            enc_gnn=CFG.GNN_TYPE,
            enc_batch_norm=1,
            pretrained=CFG.PRETRAINED,
            gpu_id=0
        )

        model = MamlMolRelationModel(trainer_args).to(device)
        trainer = MamlMolRelationTrainer(trainer_args, model)

        # Load checkpoint
        ckpt_path = args.checkpoint or os.path.join(args.output_dir, f"{CFG.MODEL_NAME.lower()}_{args.dataset}_{shot_name}_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            trainer.model.load_state_dict(ckpt['model_state'])
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: No checkpoint found. Evaluating with initialized weights.")

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

            # Adapt formats
            support_adapted = [adapt_sample_to_attfpgnn(s) for s in support_samples]
            query_adapted = [adapt_sample_to_attfpgnn(q) for q in query_samples]

            s_batch = build_attfpgnn_batch(support_adapted, device)
            q_batch = build_attfpgnn_batch(query_adapted, device)
            s_batch.smiles = [s.smiles for s in support_adapted]
            q_batch.smiles = [q.smiles for q in query_adapted]

            adapt_data = {
                's_data': s_batch, 
                's_label': s_batch.y, 
                'q_data': q_batch, 
                'q_label': q_batch.y
            }

            # Local adaptation
            local_model, out_w, out_b = trainer.adapt_few_shot(adapt_data, CFG.INNER_UPDATE_STEP * CFG.UPDATE_STEP_TEST)

            # Evaluation
            local_model.eval()
            with torch.no_grad():
                s_feats, q_feats, s_lbls = local_model(s_batch, q_batch, s_label=s_batch.y)
                q_preds = F.softmax(F.linear(q_feats, out_w, out_b), dim=1)
                q_labels = q_batch.y

                if len(torch.unique(q_labels)) > 1:
                    auroc = roc_auc_score(q_labels.cpu().numpy(), q_preds[:, 1].cpu().numpy())
                    task_results[task_name].append(auroc)

        per_task = {}
        all_task_means = []
        for task_name, aucs in task_results.items():
            m = float(np.mean(aucs))
            s = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
            per_task[task_name] = {'auc_mean': m, 'auc_std': s, 'n_episodes': len(aucs), 'raw_auc': [round(a, 4) for a in aucs]}
            all_task_means.append(m)
            print(f"  {task_name:40s}: AUC = {m:.4f} ± {s:.4f}  (n={len(aucs)})")

        om = float(np.mean(all_task_means)) if all_task_means else float('nan')
        os_ = float(np.std(all_task_means, ddof=1)) if len(all_task_means) > 1 else 0.0
        print(f"  {'Overall':40s}: AUC = {om:.4f} ± {os_:.4f}")

        all_results[shot_name] = {'auc_mean': om, 'auc_std': os_, 'per_task': per_task}

    results = {
        'model': CFG.MODEL_NAME,
        'dataset': args.dataset,
        'gnn_type': CFG.GNN_TYPE,
        'seed': args.seed,
        'meta_test_tasks': episodes.get('meta_test_tasks', []),
        'shots': all_results,
    }
    
    results_path = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Meta-test {CFG.MODEL_NAME} baseline')
    parser.add_argument('--data_dir',      type=str, required=True)
    parser.add_argument('--output_dir',    type=str, default=CFG.CHECKPOINT_DIR)
    parser.add_argument('--dataset',       type=str, default='tox21', choices=['tox21', 'sider'])
    parser.add_argument('--shots',         type=int, nargs='+', default=[5, 10])
    parser.add_argument('--checkpoint',    type=str, default=None)
    parser.add_argument('--episodes_file', type=str, required=True)
    parser.add_argument('--q_query',       type=int, default=CFG.Q_QUERY)
    parser.add_argument('--seed',          type=int, default=CFG.SEED)
    args = parser.parse_args()
    run_test(CFG, args)
