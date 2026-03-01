import os
import json
import argparse
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

TOX21_SPLITS = {
    'meta_train': [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE'
    ],
    'meta_val':  ['SR-ATAD5'],
    'meta_test': ['SR-HSE', 'SR-MMP', 'SR-p53'],
}

SIDER_SPLITS = {
    'meta_train': [
        'Infections and infestations',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'Blood and lymphatic system disorders',
        'Immune system disorders',
        'Endocrine disorders',
        'Psychiatric disorders',
        'Eye disorders',
        'Vascular disorders',
        'Respiratory, thoracic and mediastinal disorders',
        'Hepatobiliary disorders',
        'Musculoskeletal and connective tissue disorders',
        'Reproductive system and breast disorders',
        'Congenital, familial and genetic disorders',
        'General disorders and administration site conditions',
        'Investigations',
        'Surgical and medical procedures',
        'Social circumstances',
        'Product issues',
    ],
    'meta_val': [
        'Metabolism and nutrition disorders',
        'Gastrointestinal disorders',
        'Skin and subcutaneous tissue disorders',
    ],
    'meta_test': [
        'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders',
        'Cardiac disorders',
        'Nervous system disorders',
        'Injury, poisoning and procedural complications',
    ],
}

DATASET_SPLITS = {'tox21': TOX21_SPLITS, 'sider': SIDER_SPLITS}

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOANY,
]
EDGE_ATTR_DIM = len(BOND_TYPE_LIST) + len(BOND_STEREO_LIST)  # 8



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    feat = np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As',
            'Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti',
            'Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt',
            'Hg','Pb','Unknown'                       # 44
        ]) +
        one_of_k_encoding(atom.GetDegree(),          [0,1,2,3,4,5,6,7,8,9,10]) +   # 11
        one_of_k_encoding_unk(atom.GetTotalNumHs(),  [0,1,2,3,4,5,6,7,8,9,10]) +   # 11
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) + # 11
        [atom.GetIsAromatic()]                                                        # 1
    , dtype=np.float32)  # total = 78
    s = feat.sum()
    return feat / s if s != 0 else feat


def bond_features(bond):
    bt = bond.GetBondType()
    bs = bond.GetStereo()

    bond_type_enc = [int(bt == t) for t in BOND_TYPE_LIST]
    if sum(bond_type_enc) == 0:
        bond_type_enc[0] = 1   # fallback SINGLE

    stereo_enc = [int(bs == s) for s in BOND_STEREO_LIST]
    if sum(stereo_enc) == 0:
        stereo_enc[0] = 1      # fallback STEREONONE

    return bond_type_enc + stereo_enc 


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    if len(node_feats) == 0:
        return None

    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i  = bond.GetBeginAtomIdx()
        j  = bond.GetEndAtomIdx()
        ef = bond_features(bond)
        edges.append([i, j]);  edge_attrs.append(ef)   # i → j
        edges.append([j, i]);  edge_attrs.append(ef)   # j → i  (bidirectional)

    if len(edges) == 0:                               
        edges      = [[0, 0]]
        edge_attrs = [[0] * EDGE_ATTR_DIM]

    return Data(
        x          = torch.tensor(np.array(node_feats), dtype=torch.float),
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(np.array(edge_attrs), dtype=torch.float),
    )


class SMILESVocabulary:
    def __init__(self):
        self.tokens = [
            '<PAD>', '<UNK>',
            # Multi-char atoms
            'Cl','Br','Si','si','Se','se','As','as','Te','te',
            # Single-char atoms & symbols
            'C','c','N','n','O','o','S','s','P','p','F','I','H','B','b',
            '[',']','(',')','=','#','+','-',
            '1','2','3','4','5','6','7','8','9','0',
            '@','.','/', '\\','%',
        ]
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.vocab_size   = len(self.tokens)

    def tokenize(self, smiles):
        tokens, i = [], 0
        while i < len(smiles):
            if i < len(smiles) - 1:
                two = smiles[i:i+2]
                if two in self.token_to_idx:
                    tokens.append(two); i += 2; continue
            tokens.append(smiles[i]); i += 1
        return tokens

    def encode(self, smiles, max_len=200):
        tokens  = self.tokenize(smiles)
        encoded = [
            self.token_to_idx.get(t, self.token_to_idx['<UNK>'])
            for t in tokens[:max_len]
        ]
        encoded += [0] * (max_len - len(encoded))   # padding
        return torch.tensor(encoded, dtype=torch.long)


SMILES_VOCAB = SMILESVocabulary()

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp  = _FP_GEN.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float)


def process_task(df_task, task_name):
    pos_samples, neg_samples, skipped = [], [], 0

    for _, row in df_task.iterrows():
        label = row[task_name]
        if label not in [0, 1]:
            continue

        smiles = row['smiles']
        fp     = smiles_to_fingerprint(smiles)
        graph  = smiles_to_graph(smiles)
        seq    = SMILES_VOCAB.encode(smiles)

        if fp is None or graph is None or graph.x.shape[0] == 0:
            skipped += 1
            continue

        sample = {
            'fp':       fp,
            'graph':    graph,
            'sequence': seq,
            'label':    int(label),
            'smiles':   smiles,
        }
        (pos_samples if int(label) == 1 else neg_samples).append(sample)

    print(f"    {task_name}: pos={len(pos_samples)}, neg={len(neg_samples)}, "
          f"total={len(pos_samples)+len(neg_samples)}, skipped={skipped}")
    return {'pos': pos_samples, 'neg': neg_samples}


def preprocess(csv_path: str, output_dir: str, dataset: str = 'tox21'):
    if dataset not in DATASET_SPLITS:
        raise ValueError(f"Invalid: {dataset}")

    splits = DATASET_SPLITS[dataset]

    print(f"Dataset  : {dataset}")
    print(f"CSV      : {csv_path}")
    all_df = pd.read_csv(csv_path)
    print(f"Rows     : {len(all_df)}")
    print(f"EDGE_ATTR_DIM = {EDGE_ATTR_DIM}  (4 bond_type + 4 stereo)")

    os.makedirs(output_dir, exist_ok=True)
    dataset_info = {}

    for split_name, task_list in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n=== {split_name} ({len(task_list)} tasks) ===")
        dataset_info[split_name] = {}

        for task_name in task_list:
            if task_name not in all_df.columns:
                print(f"  WARNING: '{task_name}' invalid. Skipping.")
                continue

            df_task = all_df[['smiles', task_name]].dropna()
            if df_task[task_name].nunique() < 2:
                print(f"  WARNING: '{task_name}' has only 1 class. Skipping.")
                continue

            print(f"  Processing '{task_name}' ({len(df_task)} rows)...")
            task_data = process_task(df_task, task_name)

            n_pos = len(task_data['pos'])
            n_neg = len(task_data['neg'])
            if n_pos == 0 or n_neg == 0:
                print(f"  WARNING: '{task_name}' lack 1 class. Skipping.")
                continue

            safe_name = task_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            save_path = os.path.join(split_dir, f"{safe_name}.pt")
            torch.save(task_data, save_path)

            dataset_info[split_name][task_name] = {
                'n_positive': n_pos,
                'n_negative': n_neg,
                'n_total':    n_pos + n_neg,
                'save_path':  save_path,
            }
            print(f"    Saved → {save_path}")

    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"\nDataset info → {info_path}")
    print("\nPreprocessing complete!")
    return dataset_info


def load_task(task_name: str, split: str, data_dir: str) -> dict:
    safe_name = task_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
    path = os.path.join(data_dir, split, f"{safe_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not exist: {path}\n"
        )
    return torch.load(path)


def load_all_splits(data_dir: str):
    info_path = os.path.join(data_dir, 'dataset_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"dataset_info.json can't find {data_dir}\n"
        )

    with open(info_path, encoding='utf-8') as f:
        info = json.load(f)

    def _load_split(split_name):
        out = {}
        for task_name in info.get(split_name, {}):
            out[task_name] = load_task(task_name, split_name, data_dir)
            n_pos = len(out[task_name]['pos'])
            n_neg = len(out[task_name]['neg'])
            print(f"  [{split_name}] {task_name}: pos={n_pos}, neg={n_neg}")
        return out

    print(f"Loading from {data_dir} ...")
    meta_train = _load_split('meta_train')
    meta_val   = _load_split('meta_val')
    meta_test  = _load_split('meta_test')
    print(f"Done. train={len(meta_train)} tasks, "
          f"val={len(meta_val)} tasks, test={len(meta_test)} tasks.")
    return meta_train, meta_val, meta_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess molecular dataset for meta-learning')
    parser.add_argument('--csv_path',   type=str, required=True,
                        help='Path to raw CSV file')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save processed .pt files')
    parser.add_argument('--dataset',    type=str, default='tox21',
                        choices=['tox21', 'sider'],
                        help='Dataset name (determines task splits)')
    args = parser.parse_args()

    preprocess(args.csv_path, args.output_dir, args.dataset)