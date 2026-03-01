import torch
import torch.nn as nn

from BrMGD_model import (
    FingerprintMLP,
    GINEncoder,
    SequenceCNN,
    EnhancedProtoNet
)

VARIANT_NAMES = [
    'gine_only',   # GINEConv only
    'fp_only',     # Fingerprint (MLP) only
    'cnn_only',    # SMILES CNN only
    'gine_cnn',    # GINEConv + CNN
    'gine_fp',     # GINEConv + FP
    'cnn_fp',      # CNN + FP
]

VARIANT_INFO = {
    'gine_only': {'name': 'GINEConv-only',    'branches': ['GINE']},
    'fp_only':   {'name': 'Fingerprint-only', 'branches': ['FP']},
    'cnn_only':  {'name': 'CNN-only',         'branches': ['CNN']},
    'gine_cnn':  {'name': 'GINEConv + CNN',   'branches': ['GINE', 'CNN']},
    'gine_fp':   {'name': 'GINEConv + FP',    'branches': ['GINE', 'FP']},
    'cnn_fp':    {'name': 'CNN + FP',         'branches': ['CNN', 'FP']},
}


class SimpleFusion(nn.Module):
    def __init__(self, n_modalities: int = 2, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, embed_dim),
            nn.ReLU(),
        )

    def forward(self, *embeddings):
        return self.proj(torch.cat(list(embeddings), dim=-1))


class SingleEncoder(nn.Module):
    def __init__(self, branch: str, vocab_size: int = 65):
        super().__init__()
        assert branch in ('gine', 'fp', 'cnn'), \
            f"branch is 'gine' | 'fp' | 'cnn', receive: {branch}"
        self.branch = branch

        if branch == 'gine':
            self.encoder = GINEncoder()
        elif branch == 'fp':
            self.encoder = FingerprintMLP()
        else:                                      # cnn
            self.encoder = SequenceCNN(vocab_size=vocab_size)

    def forward(self, fp, graph_data, sequence):
        if self.branch == 'gine':
            return self.encoder(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                graph_data.batch,
            )
        elif self.branch == 'fp':
            return self.encoder(fp)
        else:                                      # cnn
            return self.encoder(sequence)


class DualEncoder(nn.Module):
    def __init__(self, branches: tuple, vocab_size: int = 65):
        super().__init__()
        assert len(branches) == 2, \
            f"DualEncoder ned 2 branches, {len(branches)}"
        assert len(set(branches)) == 2, \
            "2 branches must be different"
        for b in branches:
            assert b in ('gine', 'fp', 'cnn'), \
                f"branch '{b}' invalid"

        self.branches = branches

        self.encoders = nn.ModuleDict()
        for b in branches:
            if b == 'gine':
                self.encoders['gine'] = GINEncoder()
            elif b == 'fp':
                self.encoders['fp']   = FingerprintMLP()
            else:                                  # cnn
                self.encoders['cnn']  = SequenceCNN(vocab_size=vocab_size)

        self.fusion = SimpleFusion(n_modalities=2, embed_dim=128)

    def _encode_one(self, branch: str, fp, graph_data, sequence):
        if branch == 'gine':
            return self.encoders['gine'](
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                graph_data.batch,
            )
        elif branch == 'fp':
            return self.encoders['fp'](fp)
        else:                                      # cnn
            return self.encoders['cnn'](sequence)

    def forward(self, fp, graph_data, sequence):
        emb1 = self._encode_one(self.branches[0], fp, graph_data, sequence)
        emb2 = self._encode_one(self.branches[1], fp, graph_data, sequence)
        return self.fusion(emb1, emb2)

def build_model(
    variant: str,
    device: torch.device,
    vocab_size: int = 65,
) -> EnhancedProtoNet:
    if variant not in VARIANT_NAMES:
        raise ValueError(
            f"variant '{variant}' invalid.\n"
        )

 
    if variant == 'gine_only':
        encoder = SingleEncoder('gine')
    elif variant == 'fp_only':
        encoder = SingleEncoder('fp')
    elif variant == 'cnn_only':
        encoder = SingleEncoder('cnn', vocab_size=vocab_size)

    elif variant == 'gine_cnn':
        encoder = DualEncoder(('gine', 'cnn'), vocab_size=vocab_size)
    elif variant == 'gine_fp':
        encoder = DualEncoder(('gine', 'fp'))
    elif variant == 'cnn_fp':
        encoder = DualEncoder(('cnn', 'fp'), vocab_size=vocab_size)

    return EnhancedProtoNet(encoder).to(device)