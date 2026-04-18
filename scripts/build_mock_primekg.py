from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def tanimoto_neighbors(drug_table: pd.DataFrame, top_k: int) -> List[Tuple[str, str, float]]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:
        return []

    fps: Dict[str, object] = {}
    for _, row in drug_table.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        fps[str(row["drug_id"])] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    edges: List[Tuple[str, str, float]] = []
    ids = sorted(fps)
    for src in ids:
        sims = []
        for dst in ids:
            if src == dst:
                continue
            sims.append((dst, float(DataStructs.TanimotoSimilarity(fps[src], fps[dst]))))
        sims.sort(key=lambda item: item[1], reverse=True)
        for dst, score in sims[:top_k]:
            edges.append((src, dst, score))
    return edges


def build_mock_primekg(dataset: str, processed_dir: Path, max_interactions: int, top_k: int) -> None:
    source = processed_dir / f"{dataset}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Processed dataset not found: {source}")

    df = pd.read_csv(source)
    train = df[df["split"] == "train"].copy()
    drug_table = df[["drug_id", "smiles"]].drop_duplicates("drug_id").copy()
    protein_table = df[["target_id", "fasta"]].drop_duplicates("target_id").copy()

    entities = []
    for _, row in drug_table.iterrows():
        entities.append({"entity_id": f"drug:{row['drug_id']}", "name": row["drug_id"], "type": "drug", "source_id": row["drug_id"]})
    for _, row in protein_table.iterrows():
        entities.append(
            {"entity_id": f"protein:{row['target_id']}", "name": row["target_id"], "type": "protein", "source_id": row["target_id"]}
        )
    entities.extend(
        [
            {"entity_id": "disease:kinase_related", "name": "Kinase-related disease context", "type": "disease", "source_id": "mock"},
            {"entity_id": "pathway:kinase_signaling", "name": "Kinase signaling pathway", "type": "pathway", "source_id": "mock"},
        ]
    )

    edges = []
    for _, row in train.head(max_interactions).iterrows():
        edges.append(
            {
                "head": f"drug:{row['drug_id']}",
                "relation": "interacts_with",
                "tail": f"protein:{row['target_id']}",
                "weight": float(row["affinity"]),
                "evidence": f"{dataset}_train_affinity",
            }
        )

    for drug_id in drug_table["drug_id"].astype(str):
        edges.append(
            {
                "head": f"drug:{drug_id}",
                "relation": "associated_with",
                "tail": "disease:kinase_related",
                "weight": 1.0,
                "evidence": "mock_primekg_context",
            }
        )

    for target_id in protein_table["target_id"].astype(str):
        edges.append(
            {
                "head": f"protein:{target_id}",
                "relation": "participates_in",
                "tail": "pathway:kinase_signaling",
                "weight": 1.0,
                "evidence": "mock_primekg_context",
            }
        )

    for src, dst, score in tanimoto_neighbors(drug_table, top_k=top_k):
        edges.append(
            {
                "head": f"drug:{src}",
                "relation": "similar_to",
                "tail": f"drug:{dst}",
                "weight": score,
                "evidence": "morgan_tanimoto_mock_kg",
            }
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    entities_path = processed_dir / f"{dataset}_mock_primekg_entities.csv"
    edges_path = processed_dir / f"{dataset}_mock_primekg_edges.csv"
    summary_path = processed_dir / f"{dataset}_mock_primekg_summary.json"
    pd.DataFrame(entities).to_csv(entities_path, index=False)
    pd.DataFrame(edges).to_csv(edges_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "entities": len(entities),
                "edges": len(edges),
                "note": "Mock PrimeKG-style graph for Path 3 demos; not a substitute for real PrimeKG integration.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {entities_path}")
    print(f"Wrote {edges_path}")
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mock PrimeKG-style graph for Path 3 demos.")
    parser.add_argument("--dataset", choices=["davis", "kiba"], default="davis")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--max-interactions", type=int, default=5000)
    parser.add_argument("--top-k-similar", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_mock_primekg(args.dataset, args.processed_dir, args.max_interactions, args.top_k_similar)


if __name__ == "__main__":
    main()
