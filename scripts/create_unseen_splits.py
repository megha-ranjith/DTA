from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def scaffold_key(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return str(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold or Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return str(smiles)


def kmer_set(sequence: str, k: int = 3) -> set[str]:
    sequence = str(sequence)
    if len(sequence) < k:
        return {sequence}
    return {sequence[idx : idx + k] for idx in range(len(sequence) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def protein_cluster_keys(sequences: Iterable[str], threshold: float = 0.30, k: int = 3) -> Dict[str, str]:
    clusters: List[tuple[str, set[str], str]] = []
    mapping: Dict[str, str] = {}
    for seq in sorted(set(map(str, sequences)), key=lambda value: (len(value), value)):
        kmers = kmer_set(seq, k=k)
        assigned = None
        for cluster_id, centroid_kmers, _ in clusters:
            if jaccard(kmers, centroid_kmers) >= threshold:
                assigned = cluster_id
                break
        if assigned is None:
            assigned = f"protein_cluster_{len(clusters):04d}"
            clusters.append((assigned, kmers, seq))
        mapping[seq] = assigned
    return mapping


def assign_groups(df: pd.DataFrame, group_col: str, train_frac: float, val_frac: float, seed: int) -> pd.Series:
    group_sizes = df.groupby(group_col).size().reset_index(name="size")
    group_sizes = group_sizes.sample(frac=1.0, random_state=seed).sort_values("size", ascending=False)

    total = len(df)
    target_train = train_frac * total
    target_val = val_frac * total
    split_counts = {"train": 0, "val": 0, "test": 0}
    group_to_split: Dict[str, str] = {}

    for _, row in group_sizes.iterrows():
        group = row[group_col]
        size = int(row["size"])
        deficits = {
            "train": target_train - split_counts["train"],
            "val": target_val - split_counts["val"],
            "test": (total - target_train - target_val) - split_counts["test"],
        }
        split = max(deficits, key=deficits.get)
        group_to_split[group] = split
        split_counts[split] += size

    return df[group_col].map(group_to_split)


def write_summary(df: pd.DataFrame, path: Path, split_kind: str, group_col: str) -> None:
    summary = {
        "split_kind": split_kind,
        "rows": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "unique_groups": int(df[group_col].nunique()),
        "group_column": group_col,
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_scaffold_split(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float) -> pd.DataFrame:
    out = df.copy()
    out["scaffold"] = out["smiles"].map(scaffold_key)
    out["split"] = assign_groups(out, "scaffold", train_frac=train_frac, val_frac=val_frac, seed=seed)
    return out


def create_protein_cluster_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    mapping = protein_cluster_keys(out["fasta"], threshold=threshold)
    out["protein_cluster"] = out["fasta"].astype(str).map(mapping)
    out["split"] = assign_groups(out, "protein_cluster", train_frac=train_frac, val_frac=val_frac, seed=seed)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create scaffold and protein-cluster unseen split CSVs.")
    parser.add_argument("--dataset", choices=["davis", "kiba"], default="davis")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--protein-threshold", type=float, default=0.30)
    parser.add_argument("--mode", choices=["scaffold", "protein", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.processed_dir / f"{args.dataset}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Processed dataset not found: {source}")
    df = pd.read_csv(source)

    if args.mode in {"scaffold", "both"}:
        scaffold_df = create_scaffold_split(df, args.seed, args.train_frac, args.val_frac)
        out_path = args.processed_dir / f"{args.dataset}_scaffold_split.csv"
        scaffold_df.to_csv(out_path, index=False)
        write_summary(scaffold_df, out_path, "scaffold", "scaffold")
        print(f"Wrote {out_path}")

    if args.mode in {"protein", "both"}:
        protein_df = create_protein_cluster_split(
            df,
            args.seed,
            args.train_frac,
            args.val_frac,
            threshold=args.protein_threshold,
        )
        out_path = args.processed_dir / f"{args.dataset}_protein_cluster_split.csv"
        protein_df.to_csv(out_path, index=False)
        write_summary(protein_df, out_path, "protein_cluster", "protein_cluster")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
