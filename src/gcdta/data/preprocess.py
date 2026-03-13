from __future__ import annotations

import json
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .benchmarks import BENCHMARKS


def _download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        data = response.read()
    output_path.write_bytes(data)


def _load_json_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def download_raw_dataset(dataset: str, raw_root: Path, force: bool = False) -> Path:
    if dataset not in BENCHMARKS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {list(BENCHMARKS)}")

    spec = BENCHMARKS[dataset]
    dataset_raw_dir = raw_root / dataset
    dataset_raw_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, url in spec.raw_files.items():
        out_path = dataset_raw_dir / rel_path
        if out_path.exists() and not force:
            continue
        _download(url, out_path)

    return dataset_raw_dir


def _prepare_davis_or_kiba(dataset: str, raw_dir: Path, seed: int = 42) -> pd.DataFrame:
    ligands = _load_json_file(raw_dir / "ligands_can.txt")
    proteins = _load_json_file(raw_dir / "proteins.txt")

    with (raw_dir / "Y").open("rb") as f:
        affinity = pickle.load(f, encoding="latin1")
    affinity = np.asarray(affinity, dtype=np.float64)
    if dataset == "davis":
        affinity = -np.log10(affinity / 1e9)

    train_folds: List[List[int]] = _load_json_file(raw_dir / "folds" / "train_fold_setting1.txt")
    test_indices: List[int] = _load_json_file(raw_dir / "folds" / "test_fold_setting1.txt")
    test_indices_set = set(int(v) for v in test_indices)

    # Use one fold for validation and the rest for training.
    val_fold = 0
    val_indices_set = set(int(v) for v in train_folds[val_fold])
    train_indices_set = set(int(v) for fold_id, fold in enumerate(train_folds) if fold_id != val_fold for v in fold)

    ligand_ids = list(ligands.keys())
    protein_ids = list(proteins.keys())
    num_proteins = len(protein_ids)
    rows, cols = np.where(~np.isnan(affinity))

    records: List[Dict[str, object]] = []
    for row_idx, col_idx in zip(rows, cols):
        # Fold files use flat matrix indices: flat_idx = row_idx * num_proteins + col_idx
        flat_idx = int(row_idx) * num_proteins + int(col_idx)
        
        if flat_idx in train_indices_set:
            split = "train"
        elif flat_idx in val_indices_set:
            split = "val"
        elif flat_idx in test_indices_set:
            split = "test"
        else:
            # Ignore pairs not present in setting-1 split indices.
            continue

        drug_key = ligand_ids[int(row_idx)]
        target_key = protein_ids[int(col_idx)]
        records.append(
            {
                "dataset": dataset,
                "split": split,
                "drug_id": str(drug_key),
                "target_id": str(target_key),
                "smiles": str(ligands[drug_key]),
                "fasta": str(proteins[target_key]),
                "affinity": float(affinity[row_idx, col_idx]),
            }
        )

    if not records:
        raise RuntimeError(f"No usable records were produced for dataset '{dataset}'.")

    df = pd.DataFrame(records)
    # Deterministic shuffle inside each split.
    rng = np.random.default_rng(seed)
    split_frames = []
    for split_name, frame in df.groupby("split"):
        frame = frame.sample(frac=1.0, random_state=int(rng.integers(1_000_000)))
        split_frames.append(frame)
    return pd.concat(split_frames, axis=0, ignore_index=True)


def _prepare_pdbbind_v2016(raw_dir: Path, seed: int = 42) -> pd.DataFrame:
    csv_path = raw_dir / "LP_PDBBind.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required raw file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required_cols = {"header", "smiles", "seq", "value", "date", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"PDBbind source is missing required columns: {sorted(missing)}")

    df = df[df["category"].astype(str).str.lower() == "refined"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= pd.Timestamp("2016-12-31")]
    df = df.dropna(subset=["header", "smiles", "seq", "value"])
    df = df[df["smiles"].astype(str).str.len() > 0]
    df = df[df["seq"].astype(str).str.len() > 0]

    if len(df) < 20:
        raise RuntimeError(
            "Filtered PDBbind v2016 subset is unexpectedly small. "
            "Check download source in src/gcdta/data/benchmarks.py."
        )

    df = df.reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = set(order[:n_train])
    val_idx = set(order[n_train : n_train + n_val])

    split_values: List[str] = []
    for i in range(n):
        if i in train_idx:
            split_values.append("train")
        elif i in val_idx:
            split_values.append("val")
        else:
            split_values.append("test")

    out = pd.DataFrame(
        {
            "dataset": "pdbbind_v2016",
            "split": split_values,
            "drug_id": df["header"].astype(str) + "_ligand",
            "target_id": df["header"].astype(str) + "_protein",
            "smiles": df["smiles"].astype(str),
            "fasta": df["seq"].astype(str),
            "affinity": pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["affinity"]).reset_index(drop=True)
    return out


def _load_affinity_lookup(path: Path) -> Dict[str, float]:
    affinity_df = pd.read_csv(path, sep=None, engine="python")
    id_col, value_col = affinity_df.columns[:2]
    return {
        str(row[id_col]).strip(): float(row[value_col])
        for _, row in affinity_df.dropna(subset=[id_col, value_col]).iterrows()
    }


def _merge_pdbbind_split(
    split_name: str,
    smiles_path: Path,
    seq_path: Path,
    affinity_lookup: Dict[str, float],
    dataset_name: str,
) -> pd.DataFrame:
    smiles_df = pd.read_csv(smiles_path)
    seq_df = pd.read_csv(seq_path)
    if "id" in seq_df.columns and "pdbid" not in seq_df.columns:
        seq_df = seq_df.rename(columns={"id": "pdbid"})
    if "Unnamed: 0" in seq_df.columns:
        seq_df = seq_df.drop(columns=["Unnamed: 0"])
    if "pdbid" not in smiles_df.columns or "pdbid" not in seq_df.columns:
        raise RuntimeError(f"Expected 'pdbid' column in {smiles_path.name} and {seq_path.name}.")

    merged = smiles_df.merge(seq_df[["pdbid", "seq"]], on="pdbid", how="inner")
    if merged.empty:
        raise RuntimeError(f"Unable to merge {smiles_path.name} and {seq_path.name}.")

    merged["affinity"] = merged["pdbid"].astype(str).map(affinity_lookup)
    merged = merged.dropna(subset=["affinity", "smiles", "seq"]).copy()
    merged["split"] = split_name
    merged["dataset"] = dataset_name
    merged["drug_id"] = merged["pdbid"].astype(str) + "_ligand"
    merged["target_id"] = merged["pdbid"].astype(str) + "_protein"
    merged["fasta"] = merged["seq"].astype(str)
    return merged[["dataset", "split", "drug_id", "target_id", "smiles", "fasta", "affinity"]]


def _prepare_pdbbind_benchmark(dataset: str, raw_dir: Path) -> pd.DataFrame:
    affinity_lookup = _load_affinity_lookup(raw_dir / "affinity_data.csv")
    frames = [
        _merge_pdbbind_split(
            split_name="train",
            smiles_path=raw_dir / "training_smi.csv",
            seq_path=raw_dir / "training_seq_.csv",
            affinity_lookup=affinity_lookup,
            dataset_name=dataset,
        ),
        _merge_pdbbind_split(
            split_name="val",
            smiles_path=raw_dir / "validation_smi.csv",
            seq_path=raw_dir / "validation_seq_.csv",
            affinity_lookup=affinity_lookup,
            dataset_name=dataset,
        ),
    ]

    if dataset == "core2016":
        test_smi = raw_dir / "test_smi.csv"
        test_seq = raw_dir / "test_seq_.csv"
    elif dataset == "test71":
        test_smi = raw_dir / "test71_smi.csv"
        test_seq = raw_dir / "test71_seq_.csv"
    elif dataset == "test105":
        test_smi = raw_dir / "test105_smi.csv"
        test_seq = raw_dir / "test105_seq_.csv"
    else:
        raise ValueError(f"Unsupported PDBbind benchmark: {dataset}")

    frames.append(
        _merge_pdbbind_split(
            split_name="test",
            smiles_path=test_smi,
            seq_path=test_seq,
            affinity_lookup=affinity_lookup,
            dataset_name=dataset,
        )
    )
    return pd.concat(frames, axis=0, ignore_index=True)


def preprocess_dataset(dataset: str, raw_root: Path, processed_root: Path, seed: int = 42) -> Path:
    if dataset not in BENCHMARKS:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    raw_dir = raw_root / dataset
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if dataset in {"davis", "kiba"}:
        df = _prepare_davis_or_kiba(dataset=dataset, raw_dir=raw_dir, seed=seed)
    elif dataset in {"core2016", "test71", "test105"}:
        df = _prepare_pdbbind_benchmark(dataset=dataset, raw_dir=raw_dir)
    elif dataset == "pdbbind_v2016":
        df = _prepare_pdbbind_v2016(raw_dir=raw_dir, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    processed_root.mkdir(parents=True, exist_ok=True)
    output_csv = processed_root / f"{dataset}.csv"
    df.to_csv(output_csv, index=False)

    summary = {
        "dataset": dataset,
        "num_samples": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "num_unique_drugs": int(df["drug_id"].nunique()),
        "num_unique_targets": int(df["target_id"].nunique()),
        "affinity_min": float(df["affinity"].min()),
        "affinity_max": float(df["affinity"].max()),
    }
    (processed_root / f"{dataset}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_csv


def prepare_dataset(
    dataset: str,
    data_root: Path,
    force_download: bool = False,
    force_preprocess: bool = False,
    seed: int = 42,
) -> Path:
    raw_root = data_root / "raw"
    processed_root = data_root / "processed"
    processed_file = processed_root / f"{dataset}.csv"

    if force_download or not (raw_root / dataset).exists():
        download_raw_dataset(dataset=dataset, raw_root=raw_root, force=force_download)

    if force_preprocess or not processed_file.exists():
        preprocess_dataset(dataset=dataset, raw_root=raw_root, processed_root=processed_root, seed=seed)

    return processed_file


def prepare_all(data_root: Path, force_download: bool = False, force_preprocess: bool = False, seed: int = 42) -> None:
    for name in BENCHMARKS:
        prepare_dataset(
            dataset=name,
            data_root=data_root,
            force_download=force_download,
            force_preprocess=force_preprocess,
            seed=seed,
        )
