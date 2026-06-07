"""
fsgnntr_loader.py — Helpers to build FS-GNNTR compatible DataLoaders
from 3Br-MGD formatted data.

Note: These are NOT used during training (we use build_fsgnntr_batch from maml_utils).
Provided here for reference and for potential DataLoader-based evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_adapter import adapt_graph_to_fsgnntr


class FSGNNTRGraphDataset(Dataset):
    """
    Simple dataset wrapping a list of adapted FS-GNNTR samples.
    Each item: {'graph': Data(long-index), 'label': int}
    """
    def __init__(self, adapted_samples: list):
        self.samples = adapted_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def fsgnntr_collate_fn(batch_samples):
    """Custom collate for PyG Data objects."""
    graphs = []
    labels = []
    for s in batch_samples:
        g = s['graph']
        g.y = torch.tensor([s['label']], dtype=torch.float)
        graphs.append(g)
        labels.append(s['label'])
    batched = Batch.from_data_list(graphs)
    return batched, torch.tensor(labels, dtype=torch.float)


def make_fsgnntr_loader(adapted_samples: list, batch_size: int = 10, shuffle: bool = False):
    """
    Create a DataLoader for FS-GNNTR adapted samples.

    Args:
        adapted_samples : list of {'graph': Data(long-index), 'label': int}
        batch_size      : default 10 (NOTE: TR requires batch divisible by batch_size)
        shuffle         : whether to shuffle

    Returns:
        DataLoader yielding (Batch, labels) tuples
    """
    dataset = FSGNNTRGraphDataset(adapted_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=fsgnntr_collate_fn,
        num_workers=0,
        drop_last=False,
    )
