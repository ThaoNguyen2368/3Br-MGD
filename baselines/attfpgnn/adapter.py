import torch
from torch_geometric.data import Data
from baselines.graph_adapter import adapt_graph_to_fsgnntr

def adapt_sample_to_attfpgnn(sample: dict) -> Data:
    """
    Convert a 3Br-MGD sample to a PyG Data object compatible with AttFPGNN-MAML.
    AttFPGNN's encoder expects the exact same node/edge format as FS-GNNTR.
    Additionally, it requires the 'smiles' attribute for fingerprint lookup.
    """
    # Use the existing adapter to get the correct node/edge indices
    smiles = sample.get('smiles', '')
    data = adapt_graph_to_fsgnntr(sample['graph'], smiles=smiles)
    
    # Attach label and smiles
    data.y = torch.tensor([sample['label']], dtype=torch.long)
    data.smiles = sample.get('smiles', '')
    
    return data
