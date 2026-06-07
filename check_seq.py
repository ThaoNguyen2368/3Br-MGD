import torch
import sys
sys.path.append('3Br_MGD/Br_MGD')
from data import FSDataModule

dm = FSDataModule(
    data_dir='3Br_MGD/Data/tox21/processed',
    dataset='tox21',
    train_shots=5,
    test_shots=10,
    train_episodes=100,
    q_query=32,
    seed=42,
    batch_size=1
)
dm.setup()
train_loader = dm.train_dataloader()

for batch in train_loader:
    # batch: (support_fp, support_graph, support_seq, support_y, query_fp, query_graph, query_seq, query_y)
    support_seq = batch[2]
    query_seq = batch[6]
    print(f"support_seq max: {support_seq.max().item()}, min: {support_seq.min().item()}, shape: {support_seq.shape}")
    print(f"query_seq max: {query_seq.max().item()}, min: {query_seq.min().item()}, shape: {query_seq.shape}")
    break
