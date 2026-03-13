from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gcdta.data.dataset import build_split_datasets, collate_dta_batch
from gcdta.data.featurizers import protein_sequence_to_tensor, smiles_to_graph
from gcdta.data.preprocess import prepare_dataset
from gcdta.innovation_runtime import (
    PATH_SPECS,
    format_comparison_table,
    load_model_for_config,
    run_innovation_forward,
    run_innovation_paths,
    summarize_path_output,
)
from gcdta.metrics import regression_metrics
from gcdta.train_utils import set_seed, to_device


BASE_PAPER_RMSE: Dict[str, float] = {
    "davis": 0.913,
}


def build_single_batch(smiles: str, fasta: str, max_protein_len: int = 1000) -> Dict[str, Any]:
    graph: Data = smiles_to_graph(smiles)
    graph = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index.clone(),
        edge_attr=graph.edge_attr.clone(),
    )
    drug_graph = Batch.from_data_list([graph])
    target_tokens, target_physchem, target_mask = protein_sequence_to_tensor(fasta, max_len=max_protein_len)
    return {
        "drug_graph": drug_graph,
        "target_tokens": target_tokens.unsqueeze(0),
        "target_physchem": target_physchem.unsqueeze(0),
        "target_mask": target_mask.unsqueeze(0),
        "affinity": torch.tensor([0.0], dtype=torch.float32),
        "drug_node_id": torch.tensor([0], dtype=torch.long),
        "target_node_id": torch.tensor([0], dtype=torch.long),
        "smiles": [smiles],
        "fasta": [fasta],
    }


def _infer_path_metadata(config_name: Optional[str]) -> Tuple[str, str]:
    if config_name is None:
        return "base", "Baseline GCDTA"
    for path, config_file, label in PATH_SPECS:
        if config_file and config_name.endswith(config_file):
            return path, label
    return "custom", "Custom Innovation"


def print_header() -> str:
    return "\n".join(
        [
            "=" * 80,
            "GCDTA INNOVATION PREDICTION RESULTS",
            "=" * 80,
        ]
    )


def print_input_info(smiles: str, fasta: str) -> str:
    preview = fasta[:50] + ("..." if len(fasta) > 50 else "")
    return "\n".join(
        [
            f"Input Drug SMILES:           {smiles}",
            f"Input Target FASTA:          {preview}",
            f"Sequence Length:             {len(fasta)} residues",
        ]
    )


def print_baseline_prediction(summary) -> str:
    return "\n".join(
        [
            "-" * 80,
            "PRIMARY PREDICTION (BASELINE GCDTA)",
            "-" * 80,
            f"Predicted Affinity (pKd/pKi): {summary.affinity:.4f}",
            f"Processing Time:              {summary.processing_time_seconds:.4f} seconds",
        ]
    )


def print_innovation_block(path: str, label: str, summary) -> str:
    title_map = {
        "base": "[BASE] Baseline GCDTA",
        "path1": "[PATH1] Pocket Uncertainty",
        "path2": "[PATH2] Multitask Pose",
        "path3": "[PATH3] Knowledge Graph",
        "path4": "[PATH4] Structural Negatives",
    }
    lines = [
        "-" * 80,
        title_map.get(path, f"[{path.upper()}] {label}"),
        "-" * 80,
        f"Affinity:                     {summary.affinity:.4f}",
    ]
    if summary.uncertainty_variance is not None:
        lines.append(f"Variance:                     {summary.uncertainty_variance:.6f}")
    if summary.uncertainty_std is not None:
        lines.append(f"Std Dev:                      {summary.uncertainty_std:.6f}")
    if summary.uncertainty_ci_lower is not None and summary.uncertainty_ci_upper is not None:
        lines.append(
            f"95% CI:                       [{summary.uncertainty_ci_lower:.4f}, {summary.uncertainty_ci_upper:.4f}]"
        )
    if summary.pose_rmsd is not None:
        lines.append(f"Pose RMSD:                    {summary.pose_rmsd:.4f}")
    if summary.kg_adjusted_affinity is not None:
        lines.append(f"KG-Adjusted Affinity:         {summary.kg_adjusted_affinity:.4f}")
    if summary.kg_similarity is not None:
        lines.append(f"KG Similarity (Top Neighbor): {summary.kg_similarity:.4f}")
    if summary.kg_top_neighbors:
        lines.append("Top Similar Drugs:")
        for neighbor, score in summary.kg_top_neighbors[:3]:
            lines.append(f"  {neighbor} ({score:.4f})")
    if summary.structural_contrastive_loss is not None:
        lines.append(f"Structural Contrastive Loss:  {summary.structural_contrastive_loss:.6f}")
    lines.append(f"Processing Time:              {summary.processing_time_seconds:.4f} seconds")
    return "\n".join(lines)


def print_summary_table(results: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "",
            "=" * 80,
            "COMPARISON SUMMARY TABLE",
            "=" * 80,
            format_comparison_table(results),
        ]
    )


@torch.no_grad()
def compute_rmse(model_path: Path, dataset: str, device: torch.device) -> Optional[Tuple[float, float, float]]:
    paper_rmse = BASE_PAPER_RMSE.get(dataset.lower())
    if paper_rmse is None:
        return None

    processed_csv = prepare_dataset(dataset=dataset, data_root=PROJECT_ROOT / "data", seed=42)
    splits = build_split_datasets(processed_csv_path=processed_csv)
    loader = DataLoader(
        splits.test,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_dta_batch,
        num_workers=0,
    )

    model, config = load_model_for_config(model_path, None, device)
    model.eval()
    all_true = []
    all_pred = []
    for batch in loader:
        outputs = run_innovation_forward(model, to_device(batch, device), config)
        all_true.append(batch["affinity"])
        all_pred.append(outputs["affinity"].detach().cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    project_rmse = regression_metrics(y_true, y_pred)["rmse"]
    improvement = ((paper_rmse - project_rmse) / paper_rmse) * 100.0
    return paper_rmse, project_rmse, improvement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict affinity with baseline and innovation paths.")
    parser.add_argument("--smiles", type=str, required=True, help="Drug SMILES string")
    parser.add_argument("--fasta", type=str, required=True, help="Target FASTA sequence")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "results" / "best_model.pth")
    parser.add_argument("--config", type=str, help="Optional innovation config filename")
    parser.add_argument("--compare-all", action="store_true", help="Compare baseline and all innovation paths")
    parser.add_argument("--dataset", type=str, help="Optional dataset name for RMSE comparison")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Loading model: {args.model_path}")

    batch = build_single_batch(args.smiles, args.fasta)

    if args.compare_all:
        comparison_results = run_innovation_paths(args.model_path, batch, device)
        print(print_header())
        print("")
        print(print_input_info(args.smiles, args.fasta))
        print("")
        print(print_baseline_prediction(comparison_results["base"]))
        print("")

        for path in ["base", "path1", "path2", "path3", "path4"]:
            print(print_innovation_block(path, comparison_results[path].label, comparison_results[path]))
            print("")

        if args.dataset:
            rmse_values = compute_rmse(args.model_path, args.dataset, device)
            if rmse_values is not None:
                paper_rmse, project_rmse, improvement = rmse_values
                print("-" * 80)
                print("RMSE COMPARISON")
                print("-" * 80)
                print(f"BASE PAPER RMSE: {paper_rmse:.3f}")
                print(f"PROJECT RMSE:    {project_rmse:.3f}")
                print(f"IMPROVEMENT:     {improvement:.1f}%")
                print("")

        print(print_summary_table(comparison_results))

        if args.output_json:
            payload = {
                "input": {"smiles": args.smiles, "fasta": args.fasta},
                "results": {key: summary.__dict__ for key, summary in comparison_results.items()},
            }
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    path_name, label = _infer_path_metadata(args.config)
    model, config = load_model_for_config(args.model_path, args.config, device)
    model.eval()

    start = perf_counter()
    with torch.no_grad():
        outputs = run_innovation_forward(model, to_device(batch, device), config)
    elapsed = perf_counter() - start
    summary = summarize_path_output(path_name, label, outputs, elapsed)

    print(print_header())
    print("")
    print(print_input_info(args.smiles, args.fasta))
    print("")
    print(print_innovation_block(path_name, label, summary))

    if args.dataset:
        rmse_values = compute_rmse(args.model_path, args.dataset, device)
        if rmse_values is not None:
            paper_rmse, project_rmse, improvement = rmse_values
            print("")
            print("-" * 80)
            print("RMSE COMPARISON")
            print("-" * 80)
            print(f"BASE PAPER RMSE: {paper_rmse:.3f}")
            print(f"PROJECT RMSE:    {project_rmse:.3f}")
            print(f"IMPROVEMENT:     {improvement:.1f}%")

    if args.output_json:
        payload = {
            "input": {"smiles": args.smiles, "fasta": args.fasta},
            "path": path_name,
            "summary": summary.__dict__,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
