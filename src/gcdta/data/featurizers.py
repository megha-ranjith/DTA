from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


ATOM_SYMBOLS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "X",
]
FORMAL_CHARGES = [-2, -1, 0, 1, 2, 3, "other"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]
CHIRALITY_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _normalize_table(table: Dict[str, float]) -> Dict[str, float]:
    values = np.array([table[k] for k in table], dtype=np.float64)
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v == min_v:
        out = {k: 0.0 for k in table}
    else:
        out = {k: (float(v) - min_v) / (max_v - min_v) for k, v in table.items()}
    out["X"] = float(np.mean(list(out.values())))
    return out


GROUP_ALIPHATIC = {"A", "I", "L", "M", "V"}
GROUP_AROMATIC = {"F", "W", "Y"}
GROUP_POLAR_NEUTRAL = {"C", "N", "Q", "S", "T"}
GROUP_ACIDIC = {"D", "E"}
GROUP_BASIC = {"H", "K", "R"}


# 14 continuous residue properties. Together with 5 group indicators -> 19D.
RAW_RESIDUE_PROPERTIES = {
    "weight": {
        "A": 71.08,
        "C": 103.15,
        "D": 115.09,
        "E": 129.12,
        "F": 147.18,
        "G": 57.05,
        "H": 137.14,
        "I": 113.16,
        "K": 128.18,
        "L": 113.16,
        "M": 131.20,
        "N": 114.11,
        "P": 97.12,
        "Q": 128.13,
        "R": 156.19,
        "S": 87.08,
        "T": 101.11,
        "V": 99.13,
        "W": 186.22,
        "Y": 163.18,
    },
    "pka": {
        "A": 2.34,
        "C": 1.96,
        "D": 1.88,
        "E": 2.19,
        "F": 1.83,
        "G": 2.34,
        "H": 1.82,
        "I": 2.36,
        "K": 2.18,
        "L": 2.36,
        "M": 2.28,
        "N": 2.02,
        "P": 1.99,
        "Q": 2.17,
        "R": 2.17,
        "S": 2.21,
        "T": 2.09,
        "V": 2.32,
        "W": 2.83,
        "Y": 2.32,
    },
    "pkb": {
        "A": 9.69,
        "C": 10.28,
        "D": 9.60,
        "E": 9.67,
        "F": 9.13,
        "G": 9.60,
        "H": 9.17,
        "I": 9.60,
        "K": 8.95,
        "L": 9.60,
        "M": 9.21,
        "N": 8.80,
        "P": 10.60,
        "Q": 9.13,
        "R": 9.04,
        "S": 9.15,
        "T": 9.10,
        "V": 9.62,
        "W": 9.39,
        "Y": 9.62,
    },
    "pkx": {
        "A": 0.0,
        "C": 8.18,
        "D": 3.65,
        "E": 4.25,
        "F": 0.0,
        "G": 0.0,
        "H": 6.0,
        "I": 0.0,
        "K": 10.53,
        "L": 0.0,
        "M": 0.0,
        "N": 0.0,
        "P": 0.0,
        "Q": 0.0,
        "R": 12.48,
        "S": 0.0,
        "T": 0.0,
        "V": 0.0,
        "W": 0.0,
        "Y": 0.0,
    },
    "pi": {
        "A": 6.00,
        "C": 5.07,
        "D": 2.77,
        "E": 3.22,
        "F": 5.48,
        "G": 5.97,
        "H": 7.59,
        "I": 6.02,
        "K": 9.74,
        "L": 5.98,
        "M": 5.74,
        "N": 5.41,
        "P": 6.30,
        "Q": 5.65,
        "R": 10.76,
        "S": 5.68,
        "T": 5.60,
        "V": 5.96,
        "W": 5.89,
        "Y": 5.96,
    },
    "hydrophobicity_ph2": {
        "A": 47,
        "C": 52,
        "D": -18,
        "E": 8,
        "F": 92,
        "G": 0,
        "H": -42,
        "I": 100,
        "K": -37,
        "L": 100,
        "M": 74,
        "N": -41,
        "P": -46,
        "Q": -18,
        "R": -26,
        "S": -7,
        "T": 13,
        "V": 79,
        "W": 84,
        "Y": 49,
    },
    "hydrophobicity_ph7": {
        "A": 41,
        "C": 49,
        "D": -55,
        "E": -31,
        "F": 100,
        "G": 0,
        "H": 8,
        "I": 99,
        "K": -23,
        "L": 97,
        "M": 74,
        "N": -28,
        "P": -46,
        "Q": -10,
        "R": -14,
        "S": -5,
        "T": 13,
        "V": 76,
        "W": 97,
        "Y": 63,
    },
    "volume": {
        "A": 31.0,
        "C": 55.0,
        "D": 54.0,
        "E": 83.0,
        "F": 132.0,
        "G": 3.0,
        "H": 96.0,
        "I": 111.0,
        "K": 119.0,
        "L": 111.0,
        "M": 105.0,
        "N": 56.0,
        "P": 32.5,
        "Q": 85.0,
        "R": 124.0,
        "S": 32.0,
        "T": 61.0,
        "V": 84.0,
        "W": 170.0,
        "Y": 136.0,
    },
    "polarity": {
        "A": 8.1,
        "C": 5.5,
        "D": 13.0,
        "E": 12.3,
        "F": 5.2,
        "G": 9.0,
        "H": 10.4,
        "I": 5.2,
        "K": 11.3,
        "L": 4.9,
        "M": 5.7,
        "N": 11.6,
        "P": 8.0,
        "Q": 10.5,
        "R": 10.5,
        "S": 9.2,
        "T": 8.6,
        "V": 5.9,
        "W": 5.4,
        "Y": 6.2,
    },
    "polarizability": {
        "A": 0.046,
        "C": 0.128,
        "D": 0.105,
        "E": 0.151,
        "F": 0.290,
        "G": 0.000,
        "H": 0.230,
        "I": 0.186,
        "K": 0.219,
        "L": 0.186,
        "M": 0.221,
        "N": 0.134,
        "P": 0.131,
        "Q": 0.180,
        "R": 0.291,
        "S": 0.062,
        "T": 0.108,
        "V": 0.140,
        "W": 0.409,
        "Y": 0.298,
    },
    "flexibility": {
        "A": 0.357,
        "C": 0.346,
        "D": 0.511,
        "E": 0.497,
        "F": 0.314,
        "G": 0.544,
        "H": 0.323,
        "I": 0.462,
        "K": 0.466,
        "L": 0.365,
        "M": 0.295,
        "N": 0.463,
        "P": 0.509,
        "Q": 0.493,
        "R": 0.529,
        "S": 0.507,
        "T": 0.444,
        "V": 0.386,
        "W": 0.305,
        "Y": 0.420,
    },
    "helix_propensity": {
        "A": 1.42,
        "C": 0.70,
        "D": 1.01,
        "E": 1.51,
        "F": 1.13,
        "G": 0.57,
        "H": 1.00,
        "I": 1.08,
        "K": 1.16,
        "L": 1.21,
        "M": 1.45,
        "N": 0.67,
        "P": 0.57,
        "Q": 1.11,
        "R": 0.98,
        "S": 0.77,
        "T": 0.83,
        "V": 1.06,
        "W": 1.08,
        "Y": 0.69,
    },
    "sheet_propensity": {
        "A": 0.83,
        "C": 1.19,
        "D": 0.54,
        "E": 0.37,
        "F": 1.38,
        "G": 0.75,
        "H": 0.87,
        "I": 1.60,
        "K": 0.74,
        "L": 1.30,
        "M": 1.05,
        "N": 0.89,
        "P": 0.55,
        "Q": 1.10,
        "R": 0.93,
        "S": 0.75,
        "T": 1.19,
        "V": 1.70,
        "W": 1.37,
        "Y": 1.47,
    },
    "turn_propensity": {
        "A": 0.66,
        "C": 1.19,
        "D": 1.46,
        "E": 0.74,
        "F": 0.60,
        "G": 1.56,
        "H": 0.95,
        "I": 0.47,
        "K": 1.01,
        "L": 0.59,
        "M": 0.60,
        "N": 1.56,
        "P": 1.52,
        "Q": 0.98,
        "R": 0.95,
        "S": 1.43,
        "T": 0.96,
        "V": 0.50,
        "W": 0.96,
        "Y": 1.14,
    },
}


RESIDUE_PROPERTIES = {name: _normalize_table(values) for name, values in RAW_RESIDUE_PROPERTIES.items()}
PROPERTY_NAMES = list(RESIDUE_PROPERTIES.keys())


def protein_vocab_size() -> int:
    return len(AMINO_ACIDS)


def protein_physchem_dim() -> int:
    return 19


def residue_physchem_vector(residue: str) -> np.ndarray:
    residue = residue if residue in AA_TO_INDEX else "X"
    group_features = [
        1.0 if residue in GROUP_ALIPHATIC else 0.0,
        1.0 if residue in GROUP_AROMATIC else 0.0,
        1.0 if residue in GROUP_POLAR_NEUTRAL else 0.0,
        1.0 if residue in GROUP_ACIDIC else 0.0,
        1.0 if residue in GROUP_BASIC else 0.0,
    ]
    continuous_features = [RESIDUE_PROPERTIES[name][residue] for name in PROPERTY_NAMES]
    return np.asarray(group_features + continuous_features, dtype=np.float32)


def protein_sequence_to_tensor(sequence: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = (sequence or "").upper()
    seq = "".join(ch if ch in AA_TO_INDEX else "X" for ch in seq[:max_len])

    token_ids = np.zeros((max_len,), dtype=np.int64)
    physchem = np.zeros((max_len, protein_physchem_dim()), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)

    for idx, residue in enumerate(seq):
        token_ids[idx] = AA_TO_INDEX[residue]
        physchem[idx] = residue_physchem_vector(residue)
        mask[idx] = 1.0

    return torch.from_numpy(token_ids), torch.from_numpy(physchem), torch.from_numpy(mask)


def one_of_k_encoding(x: int | str | object, allowable_set: List[int | str | object]) -> List[int]:
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set")
    return [1 if x == s else 0 for s in allowable_set]


def one_of_k_encoding_unk(x: int | str | object, allowable_set: List[int | str | object]) -> List[int]:
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]


def atom_features(atom: Chem.rdchem.Atom) -> np.ndarray:
    features = (
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_of_k_encoding_unk(atom.GetDegree(), list(range(6)) + ["other"])
        + one_of_k_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGES)
        + one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATIONS)
        + [1 if atom.GetIsAromatic() else 0]
        + [1 if atom.IsInRing() else 0]
        + one_of_k_encoding_unk(atom.GetChiralTag(), CHIRALITY_TAGS)
    )
    return np.asarray(features, dtype=np.float32)


def bond_features(bond: Chem.rdchem.Bond | None) -> np.ndarray:
    if bond is None:
        return np.asarray([0.0] * (len(BOND_TYPES) + 1 + 1 + len(BOND_STEREOS)) + [1.0], dtype=np.float32)

    features = (
        one_of_k_encoding_unk(bond.GetBondType(), BOND_TYPES)
        + [1 if bond.GetIsConjugated() else 0]
        + [1 if bond.IsInRing() else 0]
        + one_of_k_encoding_unk(bond.GetStereo(), BOND_STEREOS)
        + [0]
    )
    return np.asarray(features, dtype=np.float32)


def atom_feature_dim() -> int:
    return len(ATOM_SYMBOLS) + 7 + len(FORMAL_CHARGES) + len(HYBRIDIZATIONS) + 1 + 1 + len(CHIRALITY_TAGS)


def bond_feature_dim() -> int:
    return len(BOND_TYPES) + 1 + 1 + len(BOND_STEREOS) + 1


@lru_cache(maxsize=200000)
def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor(np.stack([atom_features(atom) for atom in mol.GetAtoms()]), dtype=torch.float32)

    edges: List[Tuple[int, int]] = []
    edge_features: List[np.ndarray] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        attr = bond_features(bond)
        edges.append((begin, end))
        edges.append((end, begin))
        edge_features.append(attr)
        edge_features.append(attr)

    for atom_idx in range(mol.GetNumAtoms()):
        edges.append((atom_idx, atom_idx))
        edge_features.append(bond_features(None))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.stack(edge_features), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
