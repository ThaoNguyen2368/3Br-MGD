import os
import sys
import json
import numpy as np
import multiprocessing
import tqdm
from rdkit import Chem

_BASELINE_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROJECT_ROOT   = os.path.abspath(os.path.join(_BASELINE_ROOT, '..'))
_BRMGD_PATH     = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Br_MGD')
LOCAL_DATA_PATH = os.path.join(_BASELINE_ROOT, 'attfpgnn', 'data')

# Insert paths to import correctly
sys.path.insert(0, _BRMGD_PATH)

from data import load_all_splits
from fp_mixed import get_mixed_fps


def run_fps(smi):
    """Worker: returns (smi, fp_list) — must be top-level for multiprocessing."""
    from rdkit import Chem
    from fp_mixed import get_mixed_fps
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, None
    return smi, get_mixed_fps(mol)


def generate_fingerprints_for_dataset(dataset_name):
    """
    Generate mixed fingerprints (MACCS 167 + ErG 441 + PubChem 881 = 1489 dims)
    for all SMILES in the 3Br-MGD meta_train split and save them where
    MamlMolRelationModel expects them:
        AttFPGNN-MAML/MoleculeNet/data/all_fps.npy
        AttFPGNN-MAML/MoleculeNet/data/all_smis.list

    NOTE: Only meta_train SMILES are indexed here to maintain strict train/test
    isolation. Meta_test SMILES are reconstructed on-the-fly via
    reconstruct_sample_from_smiles() → build_smiles_lookup() during evaluation.
    """
    print(f"Generating fingerprints for {dataset_name}...")
    data_dir = os.path.join(_PROJECT_ROOT, '3Br_MGD', 'Data', dataset_name, 'processed')

    # Only load meta_train — meta_test is NOT needed for fingerprint pre-computation.
    # Each fingerprint is computed independently from SMILES via RDKit (no global
    # statistics), so excluding meta_test has zero effect on meta_train fingerprints.
    meta_train, _ = load_all_splits(data_dir)

    all_smis_set = set()
    for task_data in meta_train.values():
        # task_data = {'pos': [sample_dict, ...], 'neg': [sample_dict, ...]}
        for split_samples in task_data.values():
            for item in split_samples:
                smi = item.get('smiles') if isinstance(item, dict) else None
                if smi:
                    all_smis_set.add(smi)

    all_smis = list(all_smis_set)
    print(f"Total unique SMILES (meta_train only) for {dataset_name}: {len(all_smis)}")

    # Pre-validate SMILES before dispatching to pool
    valid_smis = []
    for smi in all_smis:
        if Chem.MolFromSmiles(smi) is not None:
            valid_smis.append(smi)
        else:
            print(f"  WARNING: Invalid SMILES skipped: {smi[:60]}")
    print(f"Valid SMILES: {len(valid_smis)}")

    # Generate fingerprints in parallel with error handling per SMILES
    fp_dict = {}
    with multiprocessing.Pool(8) as pool:
        async_results = [(smi, pool.apply_async(run_fps, args=(smi,)))
                         for smi in valid_smis]

        for smi, ar in tqdm.tqdm(async_results, desc="Calculating FPs"):
            try:
                _, fp = ar.get(timeout=30)
                if fp is not None:
                    fp_dict[smi] = np.array(fp, dtype=np.float32)
                else:
                    print(f"  WARNING: None FP for {smi[:60]}")
            except Exception as e:
                print(f"  WARNING: FP failed for {smi[:60]}: {e}")

    all_smis_final = list(fp_dict.keys())
    all_fps = np.array([fp_dict[s] for s in all_smis_final], dtype=np.float32)
    print(f"FP matrix shape: {all_fps.shape}  (expected: [{len(all_smis_final)}, 1489])")

    # Save to LOCAL_DATA_PATH where MamlMolRelationModel expects them
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
    np.save(os.path.join(LOCAL_DATA_PATH, "all_fps.npy"), all_fps)

    with open(os.path.join(LOCAL_DATA_PATH, "all_smis.list"), 'w') as fw:
        json.dump(all_smis_final, fw)

    # Dummy pharmacophore file to prevent FileNotFoundError in model
    # (USE_PHARMACOPHORE = False in repo gốc, nhưng model vẫn try load)
    pharm_path = os.path.join(LOCAL_DATA_PATH, "all_pharm_graph.npy")
    if not os.path.exists(pharm_path):
        np.save(pharm_path, np.array([]))

    print(f"Fingerprints saved to: {LOCAL_DATA_PATH}")
    print(f"  all_fps.npy   : {all_fps.shape}")
    print(f"  all_smis.list : {len(all_smis_final)} entries")
    print("Done.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['tox21', 'sider'])
    args = parser.parse_args()
    generate_fingerprints_for_dataset(args.dataset)
