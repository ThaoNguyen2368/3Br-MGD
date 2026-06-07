"""
adapter.py — Thin wrapper re-exporting graph_adapter for FS-GCvTR baseline.
FS-GCvTR uses the exact same 2-dim integer representation as FS-GNNTR.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_adapter import adapt_graph_to_fsgnntr as adapt_graph_to_fsgcvtr
from graph_adapter import adapt_sample_to_fsgnntr as adapt_sample_to_fsgcvtr

__all__ = ['adapt_graph_to_fsgcvtr', 'adapt_sample_to_fsgcvtr']
