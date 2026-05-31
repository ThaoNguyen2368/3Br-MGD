"""adapter.py — Thin re-export for GCN baseline."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_adapter import adapt_graph_to_fsgnntr, adapt_sample_to_fsgnntr
__all__ = ['adapt_graph_to_fsgnntr', 'adapt_sample_to_fsgnntr']
