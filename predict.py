from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
from torch_geometric.data import Batch, Data

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gcdta.data.featurizers import protein_sequence_to_tensor, smiles_to_graph
from gcdta.runtime import load_checkpoint
from gcdta.train_utils import to_device


def build_single_batch(smiles: str, fasta: str, max_protein_len: int) -> dict:
    graph: Data = smiles_to_graph(smiles)
    graph = Data(x=graph.x.clone(), edge_index=graph.edge_index.clone())
    drug_graph = Batch.from_data_list([graph])
    target_feat, target_mask = protein_sequence_to_tensor(fasta, max_len=max_protein_len)

    return {
        "drug_graph": drug_graph,
        "target_feat": target_feat.unsqueeze(0),
        "target_mask": target_mask.unsqueeze(0),
        "affinity": torch.tensor([0.0], dtype=torch.float32),
        "drug_node_id": torch.tensor([0], dtype=torch.long),
        "target_node_id": torch.tensor([0], dtype=torch.long),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict affinity for a single SMILES-FASTA pair.")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "results" / "best_model.pth")
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--fasta", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.model_path, device=device)

    max_len = int(config.get("max_protein_len", 1024))
    batch = build_single_batch(smiles=args.smiles, fasta=args.fasta, max_protein_len=max_len)

    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        batch = to_device(batch, device)
        pred, _ = model(batch)
    elapsed = time.perf_counter() - start
    value = float(pred.squeeze(0).detach().cpu().item())

    print("Predicting affinity for Drug-Target pair...")
    print(f"Drug SMILES: {args.smiles}")
    print(f"Target FASTA: {args.fasta}")
    print("--------------------------------------------------")
    print(f"Predicted Affinity Score (pKd/pKi): {value:.6f}")
    print(f"Processing Time: {elapsed:.4f}s")


if __name__ == "__main__":
    main()

