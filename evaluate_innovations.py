from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gcdta.config import ConfigLoader
from gcdta.data.dataset import build_split_datasets, collate_dta_batch
from gcdta.data.preprocess import prepare_dataset
from gcdta.innovation_runtime import (
    PATH_SPECS,
    evaluate_innovation_paths,
    format_comparison_table,
    load_model_for_config,
    run_innovation_forward,
    summarize_path_output,
)
from gcdta.metrics import regression_metrics
from gcdta.train_utils import save_scatter_plot, set_seed, to_device


def _infer_path_metadata(config_name: Optional[str]) -> Tuple[str, str]:
    if config_name is None:
        return "base", "Baseline"
    for path, config_file, label in PATH_SPECS:
        if config_file and config_name.endswith(config_file):
            return path, label
    return "custom", "Custom Innovation"


@torch.no_grad()
def evaluate_single_path(
    model_path: Path,
    config_name: Optional[str],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Any, Dict[str, Any]]:
    model, config = load_model_for_config(model_path, config_name, device)
    model.eval()
    all_true = []
    all_pred = []
    uncertainty_values = []
    pose_values = []
    kg_values = []
    structural_values = []
    total_time = 0.0

    for batch in loader:
        device_batch = to_device(batch, device)
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        cpu_start = perf_counter()
        if start is not None and end is not None:
            start.record()
        outputs = run_innovation_forward(model, device_batch, config)
        if start is not None and end is not None:
            end.record()
            torch.cuda.synchronize(device)
            total_time += start.elapsed_time(end) / 1000.0
        else:
            total_time += perf_counter() - cpu_start

        all_true.append(device_batch["affinity"].detach().cpu())
        all_pred.append(outputs["affinity"].detach().cpu())

        if outputs.get("uncertainty_variance") is not None:
            uncertainty_values.append(outputs["uncertainty_variance"].detach().cpu().reshape(-1))
        if outputs.get("pose_rmsd") is not None:
            pose_values.append(outputs["pose_rmsd"].detach().cpu().reshape(-1))
        if outputs.get("kg_similarity") is not None:
            kg_values.append(outputs["kg_similarity"].detach().cpu().reshape(-1))
        if outputs.get("structural_contrastive_loss") is not None:
            structural_values.append(outputs["structural_contrastive_loss"].detach().cpu().reshape(-1))

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    metrics = regression_metrics(y_true, y_pred)

    path_name, label = _infer_path_metadata(config_name)
    outputs_for_summary: Dict[str, Any] = {
        "affinity": torch.tensor(y_pred.mean()),
        "contrastive_loss": torch.tensor(0.0),
    }
    if uncertainty_values:
        outputs_for_summary["uncertainty_variance"] = torch.cat(uncertainty_values)
    if pose_values:
        outputs_for_summary["pose_rmsd"] = torch.cat(pose_values)
    if kg_values:
        outputs_for_summary["kg_similarity"] = torch.cat(kg_values)
        outputs_for_summary["kg_adjusted_affinity"] = torch.tensor(y_pred.mean())
    if structural_values:
        outputs_for_summary["structural_contrastive_loss"] = torch.cat(structural_values)

    summary = summarize_path_output(path_name, label, outputs_for_summary, total_time)
    return metrics, summary, {"y_true": y_true, "y_pred": y_pred}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and innovation paths.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained checkpoint")
    parser.add_argument("--config", type=str, help="Optional innovation config filename")
    parser.add_argument("--dataset", type=str, default="davis")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Evaluate baseline and all innovation paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config: Dict[str, Any]
    if args.config:
        config = ConfigLoader().load(args.config)
    else:
        config = {"dataset": args.dataset}

    set_seed(int(config.get("seed", args.seed)))
    output_dir = args.output_dir or args.model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_csv = prepare_dataset(
        dataset=args.dataset,
        data_root=Path(config.get("data_root", PROJECT_ROOT / "data")),
        seed=int(config.get("seed", args.seed)),
    )
    splits = build_split_datasets(processed_csv_path=processed_csv)
    test_loader = DataLoader(
        splits.test,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=False,
        collate_fn=collate_dta_batch,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare_all:
        all_results = evaluate_innovation_paths(args.model_path, test_loader, device)
        print("Evaluation comparison:")
        comparison_summaries = {path: payload["summary"] for path, payload in all_results.items()}
        print(format_comparison_table(comparison_summaries))
        print("")
        for path, payload in all_results.items():
            metrics = payload["metrics"]
            print(f"{payload['label']} ({path})")
            print(f"  CI: {metrics['ci']:.6f}")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  Pearson R: {metrics['pearson_r']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
        output = {
            path: {
                "label": payload["label"],
                "metrics": payload["metrics"],
                "summary": payload["summary"].__dict__,
            }
            for path, payload in all_results.items()
        }
        (output_dir / "evaluation_compare_all.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        return

    metrics, summary, arrays = evaluate_single_path(args.model_path, args.config, test_loader, device)
    print(f"Evaluation on {args.dataset}:")
    print(f"[Metrics]")
    print(f"  - Concordance Index (CI): {metrics['ci']:.6f}")
    print(f"  - Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print(f"  - Mean Absolute Error (MAE): {metrics['mae']:.6f}")
    print(f"  - Pearson Correlation (R): {metrics['pearson_r']:.6f}")
    print(f"  - RMSE: {metrics['rmse']:.6f}")
    print("--------------------------------------------------")
    print(format_comparison_table({summary.path: summary}))

    save_scatter_plot(
        arrays["y_true"],
        arrays["y_pred"],
        output_dir / f"{args.dataset}_innovation_scatter.png",
        title=f"{summary.label} Predicted vs Actual",
    )
    output = {
        "metrics": metrics,
        "summary": summary.__dict__,
    }
    (output_dir / "evaluation_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
