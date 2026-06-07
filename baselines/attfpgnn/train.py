"""
train.py — Meta-training entry point for AttFPGNN baseline.

Usage:
  python baselines/attfpgnn/train.py \
      --data_dir 3Br_MGD/Data/tox21/processed \
      --dataset tox21 \
      --shots 5 10
"""

import sys, os, argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy

# Path setup
_BASELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_BASELINE_ROOT, '..'))
_BRMGD_PATH    = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')
_ATTFPGNN_PATH = os.path.join(_PROJECT_ROOT, 'AttFPGNN-MAML', 'MoleculeNet')
_ADKF_IFT_PATH = os.path.join(_PROJECT_ROOT, 'AttFPGNN-MAML', 'ADKF-IFT', 'MoleculeNet')

for p in [_BRMGD_PATH, _ATTFPGNN_PATH, _ADKF_IFT_PATH, _BASELINE_ROOT, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import baselines.attfpgnn.config as CFG
from baselines.seed_utils import set_seed
from baselines.attfpgnn.adapter import adapt_sample_to_attfpgnn
from baselines.attfpgnn.prepare_fps import generate_fingerprints_for_dataset
from baselines.maml_utils import build_attfpgnn_batch

from data import load_all_splits
from BrMGD_train import create_meta_task
from maml_mol_relation_model import MamlMolRelationModel
from maml_mol_relation_trainer import MamlMolRelationTrainer
from sklearn.metrics import roc_auc_score


class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ─── Subclass: Override get_data_sample đúng cách bằng kế thừa ─────────────────
class BrMGDMamlTrainer(MamlMolRelationTrainer):
    """
    Override get_data_sample để dùng 3Br-MGD data thay vì ADKF-IFT MoleculeDataset.
    Monkey-patch không hoạt động vì train_step() gọi self.get_data_sample() bên trong class.
    Phải override đúng cách bằng kế thừa Python.

    Cũng override train_step() để fix lỗi `p_global.grad += p_local.grad`
    khi một trong hai grad là None (frozen params / unused params trong MAML).
    """

    def __init__(self, args, model, task_names, meta_train, K_shot, q_query):
        super().__init__(args, model)
        self._brmgd_task_names = task_names
        self._brmgd_meta_train = meta_train
        self._brmgd_K_shot     = K_shot
        self._brmgd_q_query    = q_query

    def get_data_sample(self, task_idx, train=True):
        """
        Override hoàn toàn: lấy dữ liệu từ 3Br-MGD và chuyển đổi sang
        AttFPGNN PyG Batch format thay vì dùng MoleculeDataset gốc.
        """
        task_name = self._brmgd_task_names[task_idx]
        task_data = self._brmgd_meta_train[task_name]

        try:
            support, query = create_meta_task(
                task_data, self._brmgd_K_shot, self._brmgd_q_query, train=True
            )
        except ValueError as e:
            raise ValueError(f"Task '{task_name}': {e}")

        # Chuyển đổi sang PyG Data objects với .smiles
        support_adapted = [adapt_sample_to_attfpgnn(s) for s in support]
        query_adapted   = [adapt_sample_to_attfpgnn(q) for q in query]

        # Dùng build_attfpgnn_batch (nhận list[Data] với .y đã set)
        s_batch = build_attfpgnn_batch(support_adapted, self.device)
        q_batch = build_attfpgnn_batch(query_adapted,   self.device)

        # Gắn smiles list vào batch để AttFPGNN tra cứu fingerprints
        s_batch.smiles = [s.smiles for s in support_adapted]
        q_batch.smiles = [q.smiles for q in query_adapted]

        adapt_data = {
            's_data':  s_batch,
            's_label': s_batch.y,
            'q_data':  q_batch,
            'q_label': q_batch.y,
            'label':   torch.cat([s_batch.y, q_batch.y], 0),
        }
        return adapt_data, {}

    def train_step(self):
        """
        Override train_step() để fix lỗi của repo gốc:
            `p_global.grad += p_local.grad`  →  crash khi grad là None.
        Dùng safe accumulation: bỏ qua params có grad=None.
        """
        import random as _random
        import numpy as _np

        self.train_epoch += 1
        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = _random.sample(task_id_list, batch_task)

        data_batches = {}
        for task_id in task_id_list:
            data_batches[task_id] = self.get_data_sample(task_id, train=True)

        for k in range(self.update_step):
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            # Khởi tạo gradient bằng 0 (giống PyTorch < 1.7)
            for p in self.model.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p.data)

            losses_eval = []

            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                local_model, output_weight, output_bias = self.adapt_few_shot(
                    train_data, inner_update_step=self.inner_update_step
                )

                support_feats, query_feats, support_labels = local_model(
                    s_data=train_data['s_data'],
                    q_data=train_data['q_data'],
                    s_label=train_data['s_label']
                )
                query_labels = train_data['q_label']
                query_preds  = F.linear(query_feats, output_weight, output_bias)
                loss = F.cross_entropy(query_preds, query_labels)

                loss.backward()

                # Cộng dồn gradient
                for p_global, p_local in zip(
                    self.model.parameters(), local_model.parameters()
                ):
                    if p_local.grad is not None:
                        p_global.grad.add_(p_local.grad)

                losses_eval.append(loss.detach().cpu().numpy())

            if self.args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_value
                )
            self.optimizer.step()

            losses_eval = _np.mean(losses_eval)
            print(
                f'Train Epoch: {self.train_epoch}, '
                f'train update step: {k}, '
                f'loss_eval: {losses_eval:.4f}'
            )

        return self.model


def run_train(CFG, args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Model   : {CFG.MODEL_NAME}")
    print(f"Dataset : {args.dataset}")
    print(f"Shots   : {args.shots}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Tự động sinh fingerprints nếu chưa có
    fps_path = os.path.join(CFG.ATTFPGNN_DATA_DIR, "all_fps.npy")
    if not os.path.exists(fps_path):
        print("Fingerprints not found. Generating...")
        generate_fingerprints_for_dataset(args.dataset)

    # Cập nhật module-level globals trong maml_mol_relation_model nếu chúng còn rỗng
    # (được load khi import, nhưng file chưa tồn tại lúc đó)
    # KHÔNG dùng importlib.reload() vì sẽ tạo class mới → super() TypeError
    import maml_mol_relation_model as _mmrm
    import json as _json
    if len(_mmrm.all_smis) == 0:
        import numpy as _np
        _mmrm.all_fps  = _np.load(os.path.join(CFG.ATTFPGNN_DATA_DIR, "all_fps.npy"))
        with open(os.path.join(CFG.ATTFPGNN_DATA_DIR, "all_smis.list")) as _fr:
            _mmrm.all_smis = _json.load(_fr)
        _mmrm.smi2id   = {smi: idx for idx, smi in enumerate(_mmrm.all_smis)}
        print(f"  Loaded {len(_mmrm.all_smis)} SMILES into maml_mol_relation_model globals.")

    meta_train, _ = load_all_splits(args.data_dir)
    task_names = list(meta_train.keys())
    print(f"Meta-train tasks ({len(task_names)}): {task_names}")

    for K_shot in args.shots:
        shot_name = f"{K_shot}-shot"
        print(f"\n{'='*30} {shot_name.upper()} {'='*30}")
        set_seed(args.seed)

        # Tạo DummyArgs cho MamlMolRelationModel và Trainer
        trainer_args = DummyArgs(
            device=device,
            dataset=args.dataset,
            test_dataset=args.dataset,
            data_dir=CFG.ATTFPGNN_DATA_DIR,
            train_tasks=list(range(len(task_names))),
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
            # Cần khai báo để tránh AttributeError nếu PRETRAINED=True
            pretrained_weight_path=os.path.join(
                _ATTFPGNN_PATH, 'chem_lib', 'models', 'weights',
                'gin_supervised_contextpred.pth'
            ),
            gpu_id=0
        )

        model = MamlMolRelationModel(trainer_args).to(device)

        # Dùng BrMGDMamlTrainer (subclass) thay vì MamlMolRelationTrainer gốc
        # để get_data_sample được override đúng cách (Python inheritance, không monkey-patch)
        trainer = BrMGDMamlTrainer(
            args=trainer_args,
            model=model,
            task_names=task_names,
            meta_train=meta_train,
            K_shot=K_shot,
            q_query=args.q_query,
        )
        trainer.train_tasks = list(range(len(task_names)))

        best_auroc = 0.0
        best_state = None
        patience_ctr = 0

        for epoch in range(1, args.max_epochs + 1):
            trainer.model.train()
            # train_step() bên trong gọi self.get_data_sample() → sẽ dùng
            # BrMGDMamlTrainer.get_data_sample() đúng cách nhờ kế thừa
            trainer.train_step()

            # Evaluation nhanh trên một số episode để theo dõi / early stopping
            trainer.model.eval()
            epoch_aurocs = []
            n_eval = max(1, args.train_episodes // 10)
            with torch.no_grad():
                for _ in range(n_eval):
                    task_id = random.choice(range(len(task_names)))
                    try:
                        train_data, _ = trainer.get_data_sample(task_id, train=True)
                    except ValueError:
                        continue
                    local_model, out_w, out_b = trainer.adapt_few_shot(
                        train_data, trainer_args.inner_update_step
                    )
                    local_model.eval()
                    _, q_feats, _ = local_model(
                        train_data['s_data'], train_data['q_data'], s_label=train_data['s_label']
                    )
                    q_preds  = F.softmax(F.linear(q_feats, out_w, out_b), dim=1)
                    q_labels = train_data['q_label']
                    if len(torch.unique(q_labels)) > 1:
                        auc = roc_auc_score(
                            q_labels.cpu().numpy(),
                            q_preds[:, 1].cpu().numpy()
                        )
                        epoch_aurocs.append(auc)

            epoch_auroc = np.mean(epoch_aurocs) if epoch_aurocs else float('nan')

            if not np.isnan(epoch_auroc) and epoch_auroc > (best_auroc + 0.001):
                best_auroc = epoch_auroc
                best_state = deepcopy(trainer.model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: eval_auc={epoch_auroc:.4f}, patience={patience_ctr}/{args.patience}")

            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}. Best AUC = {best_auroc:.4f}")
                break

        # Lưu checkpoint tốt nhất
        if best_state is not None:
            ckpt = {'model_state': best_state, 'val_auroc': best_auroc, 'config': vars(args)}
            ckpt_path = os.path.join(
                args.output_dir,
                f"{CFG.MODEL_NAME.lower()}_{args.dataset}_{shot_name}_best.pt"
            )
            torch.save(ckpt, ckpt_path)
            print(f"  Best checkpoint → {ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'Meta-train {CFG.MODEL_NAME} baseline')
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--output_dir',     type=str, default=CFG.CHECKPOINT_DIR)
    parser.add_argument('--dataset',        type=str, default='tox21', choices=['tox21', 'sider'])
    parser.add_argument('--shots',          type=int, nargs='+', default=[5, 10])
    parser.add_argument('--max_epochs',     type=int, default=CFG.MAX_EPOCHS)
    parser.add_argument('--patience',       type=int, default=CFG.PATIENCE)
    parser.add_argument('--train_episodes', type=int, default=CFG.BATCH_TASK * 10)
    parser.add_argument('--q_query',        type=int, default=CFG.Q_QUERY)
    parser.add_argument('--seed',           type=int, default=CFG.SEED)
    args = parser.parse_args()
    run_train(CFG, args)
