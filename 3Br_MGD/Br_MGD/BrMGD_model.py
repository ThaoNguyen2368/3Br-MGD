import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class FingerprintMLP(nn.Module):
    """
    MLP encoder for Morgan fingerprint.
    Input : [B, 2048]
    Output: [B, 128]
    """
    def __init__(self, in_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.fc(x)


class GINEncoder(nn.Module):
    """
    2-layer GINEConv encoder for molecular graph.
    Input : x=[n_atoms, 78], edge_index=[2, n_edges], edge_attr=[n_edges, 8]
    Output: [B, 128]
    """
    def __init__(self, in_dim: int = 78, edge_dim: int = 8, out_dim: int = 128):
        super().__init__()

        self.conv1 = GINEConv(
            nn=nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU()),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            nn=nn.Sequential(nn.Linear(32, 64), nn.ReLU()),
            edge_dim=edge_dim,
        )
        self.lin = nn.Linear(64, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class SequenceCNN(nn.Module):
    """
    Multi-scale 1D CNN encoder for SMILES sequence.
    Input : [B, 200]  (token indices)
    Output: [B, 128]
    """
    def __init__(self, vocab_size: int = None, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()

        if vocab_size is None:
            try:
                from data import SMILES_VOCAB
                vocab_size = SMILES_VOCAB.vocab_size
            except ImportError:
                vocab_size = 65   # default

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        #  kernel 3, 5, 7 → concat [B, 192, L]
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2),
            nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3),
        ])

        self.conv_post = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        # x: [B, 200]
        x = self.embedding(x)          # [B, 200, embed_dim]
        x = x.transpose(1, 2)          # [B, embed_dim, 200]

        branches = [F.relu(conv(x)) for conv in self.conv_branches]
        x = torch.cat(branches, dim=1) # [B, 192, 200]
        x = self.conv_post(x)          # [B, 256] → Flatten
        return self.fc(x)              # [B, out_dim]


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate_with_ctx = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),   # 4 = 3 modality + task_ctx
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )
        self.gate_no_ctx = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),   # 3 modality, no task_ctx
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

        self.cnn_residual_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, fp_emb, gnn_emb, seq_emb, task_ctx=None):
        B = fp_emb.size(0)

        tokens = torch.stack([fp_emb, gnn_emb, seq_emb], dim=1)  # [B, 3, 128]
        attn_out, attn_weights = self.self_attn(tokens, tokens, tokens)
        attn_out = self.norm(tokens + self.dropout(attn_out))      # [B, 3, 128]

        # --- Task-conditioned gate ---
        if task_ctx is not None:
            # task_ctx: [128] → broadcast sang [B, 128]
            task_ctx_exp = task_ctx.unsqueeze(0).expand(B, -1)     # [B, 128]
            global_ctx   = torch.cat(
                [fp_emb, gnn_emb, seq_emb, task_ctx_exp], dim=-1   # [B, 512]
            )
            gates = self.gate_with_ctx(global_ctx)                  # [B, 3]
        else:
            global_ctx = torch.cat([fp_emb, gnn_emb, seq_emb], dim=-1)  # [B, 384]
            gates      = self.gate_no_ctx(global_ctx)                    # [B, 3]

        # --- Weighted sum: [B, 3, 1] * [B, 3, 128] → [B, 128] ---
        fused = (attn_out * gates.unsqueeze(-1)).sum(dim=1)        # [B, 128]

        w = self.cnn_residual_weight.clamp(0.0, 1.0)
        fused = fused + w * seq_emb                                # [B, 128]

        return fused, gates, attn_weights


class TripleEncoder(nn.Module):
    def __init__(self, vocab_size: int = None):
        super().__init__()
        self.fp_encoder  = FingerprintMLP()
        self.gnn_encoder = GINEncoder()
        self.seq_encoder = SequenceCNN(vocab_size=vocab_size)
        self.fusion      = AttentionFusion(embed_dim=128, num_heads=4, dropout=0.1)

    def forward(self, fp, graph_data, sequence, task_ctx=None):
        fp_emb  = self.fp_encoder(fp)
        gnn_emb = self.gnn_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
            graph_data.batch,
        )
        seq_emb = self.seq_encoder(sequence)

        fused, _, _ = self.fusion(fp_emb, gnn_emb, seq_emb, task_ctx=task_ctx)
        return fused

    def forward_with_attention(self, fp, graph_data, sequence, task_ctx=None):
        fp_emb  = self.fp_encoder(fp)
        gnn_emb = self.gnn_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
            graph_data.batch,
        )
        seq_emb = self.seq_encoder(sequence)

        fused, gates, attn_weights = self.fusion(
            fp_emb, gnn_emb, seq_emb, task_ctx=task_ctx
        )

        return fused, {
            'attn_weights':  attn_weights,
            'gates':         gates,
            'cnn_residual':  self.fusion.cnn_residual_weight.item(),
            'fp_emb':        fp_emb,
            'gnn_emb':       gnn_emb,
            'seq_emb':       seq_emb,
        }

def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, m = x.size(0), y.size(0)
    x = x.unsqueeze(1).expand(n, m, -1)
    y = y.unsqueeze(0).expand(n, m, -1)
    return ((x - y) ** 2).sum(dim=2)


class EnhancedProtoNet(nn.Module):
    def __init__(self, encoder: TripleEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        support_fp, support_graph, support_seq, support_y,
        query_fp,   query_graph,   query_seq,
    ):
        support_emb = self.encoder(
            support_fp, support_graph, support_seq, task_ctx=None
        )                                                   # [S, 128]

        task_ctx = support_emb.mean(dim=0)                  # [128]

        query_emb = self.encoder(
            query_fp, query_graph, query_seq, task_ctx=task_ctx
        )                                                   # [Q, 128]

        classes      = torch.unique(support_y)
        class_to_idx = {c.item(): i for i, c in enumerate(classes)}

        prototypes = torch.stack([
            support_emb[support_y == c].mean(0)
            for c in classes
        ])                                                  # [N_way, 128]

        dists  = pairwise_dist(query_emb, prototypes)       # [Q, N_way]
        logits = -dists

        return logits, class_to_idx