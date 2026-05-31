"""
smoke_test.py — Verify all baseline imports work correctly.
Run from D:\\3Br_MGD:
    conda run -n 3Br_MGD python baselines/smoke_test.py
"""
import sys
import os

# Path setup — ORDER MATTERS: 3Br-MGD MUST come before FS-GNNTR to avoid 'data' module conflict
# FS-GNNTR also has a data.py; putting it first would shadow 3Br_MGD's data.py
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_BASELINE_ROOT = os.path.abspath(os.path.dirname(__file__))
_BRMGD_PATH = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')
_FSGNNTR_PATH = os.path.join(_PROJECT_ROOT, 'FS-GNNTR_repo', 'FS-GNNTR')

# Insert in REVERSE priority order (last insert = highest priority)
# Final priority: BASELINE > BRMGD > FSGNNTR
for p in [_FSGNNTR_PATH, _BRMGD_PATH, _BASELINE_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

errors = []

def check(name, fn):
    try:
        fn()
        print(f"  [OK]  {name}")
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
        errors.append((name, str(e)))

print("=" * 60)
print("SMOKE TEST — Baseline Framework")
print("=" * 60)

# 1. seed_utils
def test_seed():
    from seed_utils import set_seed
    set_seed(42)
check("seed_utils.set_seed", test_seed)

# 2. graph_adapter
def test_adapter():
    from graph_adapter import adapt_graph_to_fsgnntr, adapt_sample_to_fsgnntr, ATOM_3BR_TO_FSGNNTR
    assert ATOM_3BR_TO_FSGNNTR.shape[0] == 44, f"Expected 44, got {ATOM_3BR_TO_FSGNNTR.shape[0]}"
    print(f"         ATOM_3BR_TO_FSGNNTR: shape={ATOM_3BR_TO_FSGNNTR.shape}, dtype={ATOM_3BR_TO_FSGNNTR.dtype}")
check("graph_adapter", test_adapter)

# 3. episode_manager
def test_ep_manager():
    from episode_manager import load_episodes, generate_episodes, reconstruct_sample_from_smiles
check("episode_manager", test_ep_manager)

# 4. 3Br-MGD data pipeline
def test_data():
    from data import load_all_splits, smiles_to_graph
check("3Br-MGD data module", test_data)

# 5. BrMGD_train
def test_brmgd_train():
    from BrMGD_train import create_meta_task
check("BrMGD_train.create_meta_task", test_brmgd_train)

# 6. FS-GNNTR transformer
def test_transformer():
    from transformer import GNN_prediction
check("FS-GNNTR transformer.GNN_prediction", test_transformer)

# 7. maml_utils
def test_maml():
    from maml_utils import (
        build_fsgnntr_batch, compute_gnn_loss, compute_gnn_auroc,
        maml_inner_update, meta_train_step_gnn, meta_test_step_gnn,
    )
check("maml_utils", test_maml)

# 8. gnn_baseline_runner
def test_runner():
    from gnn_baseline_runner import build_gnn_model, run_train, run_test
check("gnn_baseline_runner", test_runner)

# 9. Load tox21 splits and check meta_test tasks
def test_load_splits():
    from data import load_all_splits
    data_dir = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Data', 'tox21', 'processed')
    meta_train, meta_test = load_all_splits(data_dir)
    print(f"         meta_train tasks: {list(meta_train.keys())}")
    print(f"         meta_test tasks : {list(meta_test.keys())}")
    assert len(meta_test) > 0, "No meta_test tasks found!"
check("load_all_splits(tox21)", test_load_splits)

# 10. Graph adapter on a real molecule
def test_graph_adapter_real():
    from data import smiles_to_graph
    from graph_adapter import adapt_graph_to_fsgnntr
    import torch
    smiles = "CCO"  # Ethanol
    graph = smiles_to_graph(smiles)
    assert graph is not None, "smiles_to_graph returned None for CCO"
    adapted = adapt_graph_to_fsgnntr(graph)
    assert adapted.x.dtype == torch.long, f"x dtype should be long, got {adapted.x.dtype}"
    assert adapted.x.shape[1] == 2, f"x shape[1] should be 2, got {adapted.x.shape[1]}"
    assert adapted.edge_attr.dtype == torch.long, f"edge_attr dtype should be long"
    assert adapted.edge_attr.shape[1] == 2, f"edge_attr shape[1] should be 2"
    print(f"         CCO: x={adapted.x.shape}, edge_attr={adapted.edge_attr.shape}, edge_index={adapted.edge_index.shape}")
check("graph_adapter on real SMILES (CCO)", test_graph_adapter_real)

# 11. Check FS-GNNTR baselines sub-packages
def test_fsgnntr_pkg():
    sys.path.insert(0, os.path.join(_BASELINE_ROOT, 'fsgnntr'))
    import fsgnntr.config as C
    print(f"         FS-GNNTR config: EMB={C.EMB_SIZE}, LAYERS={C.GRAPH_LAYERS}, GNN={C.GNN_TYPE}")
check("baselines/fsgnntr config", test_fsgnntr_pkg)

def test_gcn_pkg():
    sys.path.insert(0, os.path.join(_BASELINE_ROOT, 'gcn'))
    import gcn.config as C
    print(f"         GCN config: EMB={C.EMB_SIZE}, GNN={C.GNN_TYPE}")
check("baselines/gcn config", test_gcn_pkg)

def test_gin_pkg():
    sys.path.insert(0, os.path.join(_BASELINE_ROOT, 'gin'))
    import gin.config as C
    print(f"         GIN config: EMB={C.EMB_SIZE}, GNN={C.GNN_TYPE}")
check("baselines/gin config", test_gin_pkg)

def test_graphsage_pkg():
    sys.path.insert(0, os.path.join(_BASELINE_ROOT, 'graphsage'))
    import graphsage.config as C
    print(f"         GraphSAGE config: EMB={C.EMB_SIZE}, GNN={C.GNN_TYPE}")
check("baselines/graphsage config", test_graphsage_pkg)

print()
print("=" * 60)
if errors:
    print(f"RESULT: {len(errors)} ERROR(S) found:")
    for name, msg in errors:
        print(f"  ✗ {name}: {msg}")
    sys.exit(1)
else:
    print(f"RESULT: ALL {11} CHECKS PASSED ✓")
    print("=" * 60)
