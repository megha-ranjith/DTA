from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .featurizers import protein_sequence_to_tensor, smiles_to_graph


@dataclass
class SplitDatasets:
    train: "DTADataset"
    val: "DTADataset"
    test: "DTADataset"
    drug_id_to_index: Dict[str, int]
    target_id_to_index: Dict[str, int]


class DTADataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        drug_id_to_index: Dict[str, int],
        target_id_to_index: Dict[str, int],
        max_protein_len: int = 1000,
    ) -> None:
        required_cols = {"drug_id", "target_id", "smiles", "fasta", "affinity"}
        missing = required_cols - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing required dataset columns: {sorted(missing)}")

        self.df = dataframe.reset_index(drop=True).copy()
        self.drug_id_to_index = drug_id_to_index
        self.target_id_to_index = target_id_to_index
        self.max_protein_len = max_protein_len

        self._target_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _get_target_features(self, fasta: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fasta not in self._target_cache:
            self._target_cache[fasta] = protein_sequence_to_tensor(fasta, max_len=self.max_protein_len)
        token_ids, physchem, mask = self._target_cache[fasta]
        return token_ids.clone(), physchem.clone(), mask.clone()

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        drug_id = str(row["drug_id"])
        target_id = str(row["target_id"])
        smiles = str(row["smiles"])
        fasta = str(row["fasta"])
        affinity = float(row["affinity"])

        graph: Data = smiles_to_graph(smiles)
        graph = Data(
            x=graph.x.clone(),
            edge_index=graph.edge_index.clone(),
            edge_attr=graph.edge_attr.clone(),
        )
        target_tokens, target_physchem, target_mask = self._get_target_features(fasta)

        return {
            "drug_graph": graph,
            "target_tokens": target_tokens,
            "target_physchem": target_physchem,
            "target_mask": target_mask,
            "affinity": affinity,
            "drug_node_id": self.drug_id_to_index[drug_id],
            "target_node_id": self.target_id_to_index[target_id],
            "smiles": smiles,
            "fasta": fasta,
        }


def collate_dta_batch(samples: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    drug_graphs = [sample["drug_graph"] for sample in samples]
    drug_batch = Batch.from_data_list(drug_graphs)

    target_tokens = torch.stack([sample["target_tokens"] for sample in samples], dim=0)
    target_physchem = torch.stack([sample["target_physchem"] for sample in samples], dim=0)
    target_mask = torch.stack([sample["target_mask"] for sample in samples], dim=0)
    affinity = torch.tensor([sample["affinity"] for sample in samples], dtype=torch.float32)
    drug_node_id = torch.tensor([sample["drug_node_id"] for sample in samples], dtype=torch.long)
    target_node_id = torch.tensor([sample["target_node_id"] for sample in samples], dtype=torch.long)

    return {
        "drug_graph": drug_batch,
        "target_tokens": target_tokens,
        "target_physchem": target_physchem,
        "target_mask": target_mask,
        "affinity": affinity,
        "drug_node_id": drug_node_id,
        "target_node_id": target_node_id,
        "smiles": [str(sample["smiles"]) for sample in samples],
        "fasta": [str(sample["fasta"]) for sample in samples],
    }


def _build_node_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    drug_ids = sorted(df["drug_id"].astype(str).unique().tolist())
    target_ids = sorted(df["target_id"].astype(str).unique().tolist())
    drug_map = {key: idx for idx, key in enumerate(drug_ids)}
    target_map = {key: idx for idx, key in enumerate(target_ids)}
    return drug_map, target_map


def build_split_datasets(processed_csv_path: Path, max_protein_len: int = 1000) -> SplitDatasets:
    df = pd.read_csv(processed_csv_path)
    if "split" not in df.columns:
        raise ValueError("Processed dataset must include a 'split' column.")

    drug_map, target_map = _build_node_maps(df)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Invalid split sizes in {processed_csv_path}. "
            "Expected non-empty train/val/test splits."
        )

    return SplitDatasets(
        train=DTADataset(train_df, drug_map, target_map, max_protein_len=max_protein_len),
        val=DTADataset(val_df, drug_map, target_map, max_protein_len=max_protein_len),
        test=DTADataset(test_df, drug_map, target_map, max_protein_len=max_protein_len),
        drug_id_to_index=drug_map,
        target_id_to_index=target_map,
    )
