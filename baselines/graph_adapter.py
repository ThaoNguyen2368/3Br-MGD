"""
graph_adapter.py — Converts 3Br-MGD float one-hot graphs to FS-GNNTR long-index format.

3Br-MGD graph:
    x          : [N, 78]  float32  (one-hot: 44 atom + 11 degree + 11 H + 11 valence + 1 aromatic)
    edge_attr  : [E, 8]   float32  (one-hot: 4 bond_type + 4 stereo)

FS-GNNTR graph (expected by GNN.forward):
    x          : [N, 2]   long     (atom_type_idx in range(1,119), chirality_idx in {0,1,2,3})
    edge_attr  : [E, 2]   long     (bond_type_idx in {0,1,2,3}, bond_dir_idx in {0,1,2})

DOES NOT MODIFY any model architecture. Only transforms data format.
"""

import torch
from torch_geometric.data import Data

# ─── Atom Symbol → FS-GNNTR index mapping ─────────────────────────────────────
# 3Br-MGD one-hot atom list (44 entries, positions 0..43)
ATOM_SYMBOLS_3BR = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
    'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
    'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
]

# FS-GNNTR: possible_atomic_num_list = list(range(1, 119))
# Index = atomic_number - 1  (range 0..117)
# For 'Unknown' / out-of-range: use 118 (within the 120-size embedding)
_SYMBOL_TO_ATOMIC_NUM = {
    'C': 6,  'N': 7,  'O': 8,  'S': 16, 'F': 9,  'Si': 14, 'P': 15,
    'Cl': 17, 'Br': 35, 'Mg': 12, 'Na': 11, 'Ca': 20, 'Fe': 26,
    'As': 33, 'Al': 13, 'I': 53,  'B': 5,  'V': 23, 'K': 19, 'Tl': 81,
    'Yb': 70, 'Sb': 51, 'Sn': 50, 'Ag': 47, 'Pd': 46, 'Co': 27, 'Se': 34,
    'Ti': 22, 'Zn': 30, 'H': 1,  'Li': 3,  'Ge': 32, 'Cu': 29, 'Au': 79,
    'Ni': 28, 'Cd': 48, 'In': 49, 'Mn': 25, 'Zr': 40, 'Cr': 24, 'Pt': 78,
    'Hg': 80, 'Pb': 82, 'Unknown': None,
}

# Precomputed lookup table: 3Br-MGD atom index (0..43) → FS-GNNTR atom_type index (0..118)
_ATOM_3BR_TO_FSGNNTR = []
for _sym in ATOM_SYMBOLS_3BR:
    _anum = _SYMBOL_TO_ATOMIC_NUM.get(_sym)
    if _anum is None or _anum < 1 or _anum > 118:
        _ATOM_3BR_TO_FSGNNTR.append(118)   # fallback: extra mask token slot
    else:
        _ATOM_3BR_TO_FSGNNTR.append(_anum - 1)   # 0-indexed
ATOM_3BR_TO_FSGNNTR = torch.tensor(_ATOM_3BR_TO_FSGNNTR, dtype=torch.long)

# ─── Bond type mapping ─────────────────────────────────────────────────────────
# 3Br-MGD edge_attr[:, :4]: [SINGLE, DOUBLE, TRIPLE, AROMATIC] → indices 0,1,2,3
# FS-GNNTR possible_bonds:  [SINGLE, DOUBLE, TRIPLE, AROMATIC] → indices 0,1,2,3
# → Direct argmax mapping, NO remapping needed.

# Bond direction: 3Br-MGD stores stereo (not direction).
# FS-GNNTR possible_bond_dirs: [NONE, ENDUPRIGHT, ENDDOWNRIGHT] → default 0 (NONE)


# ─── Format-conversion cache (keyed by SMILES string for GC safety) ──────────
# Using SMILES as key instead of id(graph) because Python's id() returns a memory
# address that can be reused after garbage collection, potentially returning the
# wrong cached graph for a new object at the same address.
_FSGNNTR_CACHE: dict = {}   # Dict[smiles: str -> Data(long-index format)]


def clear_fsgnntr_cache() -> None:
    """
    Clear the format-conversion cache.
    Call this between different model runs within the same process to prevent
    stale cached graphs from persisting across experiments.

    Example:
        from graph_adapter import clear_fsgnntr_cache
        clear_fsgnntr_cache()   # reset before each baseline model
    """
    _FSGNNTR_CACHE.clear()


def adapt_graph_to_fsgnntr(graph_3br: Data, smiles: str = '') -> Data:
    """
    Convert a 3Br-MGD molecular graph to FS-GNNTR long-index format.

    Args:
        graph_3br : Source graph in 3Br-MGD float one-hot format.
        smiles    : SMILES string of the molecule (used as cache key).
                    If empty, caching is skipped and conversion runs every call.
    """
    if smiles and smiles in _FSGNNTR_CACHE:
        cached = _FSGNNTR_CACHE[smiles]
        return Data(x=cached.x, edge_index=cached.edge_index, edge_attr=cached.edge_attr)

    N = graph_3br.x.shape[0]
    E = graph_3br.edge_attr.shape[0]

    # ── Node features ──────────────────────────────────────────────────────────
    # Recover atom 3Br-index from one-hot (first 44 dims)
    atom_oh = graph_3br.x[:, :44]                          # [N, 44]
    atom_3br_idx = atom_oh.argmax(dim=1).long()            # [N]  0..43
    # Map to FS-GNNTR atomic-number index via precomputed table
    lookup = ATOM_3BR_TO_FSGNNTR.to(atom_3br_idx.device)
    atom_type_idx = lookup[atom_3br_idx]                   # [N]
    chirality_idx = torch.zeros(N, dtype=torch.long,
                                device=graph_3br.x.device) # default: CHI_UNSPECIFIED (0)
    x_new = torch.stack([atom_type_idx, chirality_idx], dim=1)  # [N, 2]

    # ── Edge features ───────────────────────────────────────────────────────────
    # Bond type from one-hot (first 4 dims of edge_attr)
    bond_oh = graph_3br.edge_attr[:, :4]                   # [E, 4]
    bond_type_idx = bond_oh.argmax(dim=1).long()           # [E]  0=SINGLE..3=AROMATIC
    bond_dir_idx = torch.zeros(E, dtype=torch.long,
                               device=graph_3br.edge_attr.device)  # default: NONE (0)
    edge_attr_new = torch.stack([bond_type_idx, bond_dir_idx], dim=1)  # [E, 2]

    new_graph = Data(
        x=x_new,
        edge_index=graph_3br.edge_index.clone(),
        edge_attr=edge_attr_new,
    )

    if smiles:
        _FSGNNTR_CACHE[smiles] = new_graph

    return Data(x=new_graph.x, edge_index=new_graph.edge_index, edge_attr=new_graph.edge_attr)


def adapt_sample_to_fsgnntr(sample: dict) -> dict:
    """
    Adapt one 3Br-MGD sample dict so the 'graph' field uses FS-GNNTR format.
    Preserves 'label' and 'smiles'; ignores 'fp' and 'sequence'.

    Returns minimal dict: {'graph': Data(long-index format), 'label': int, 'smiles': str}
    """
    smiles = sample.get('smiles', '')
    return {
        'graph':  adapt_graph_to_fsgnntr(sample['graph'], smiles=smiles),
        'label':  sample['label'],
        'smiles': smiles,
    }
