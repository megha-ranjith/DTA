from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gcdta.data.dataset import build_split_datasets, collate_dta_batch
from gcdta.data.preprocess import prepare_dataset
from gcdta.metrics import regression_metrics
from gcdta.runtime import load_checkpoint
from gcdta.train_utils import ensure_dir, save_scatter_plot, set_seed, to_device


DATASET_CHOICES = ["davis", "kiba", "core2016", "test71", "test105", "pdbbind_v2016"]


@torch.no_grad()
def test_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    criterion = nn.MSELoss()
    all_true = []
    all_pred = []
    running = 0.0
    count = 0

    for batch in loader:
        batch = to_device(batch, device)
        pred, _ = model(batch)
        loss = criterion(pred, batch["affinity"])
        bs = int(batch["affinity"].shape[0])
        running += float(loss.item()) * bs
        count += bs
        all_true.append(batch["affinity"].detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred, running / max(count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GCDTA model on a dataset split.")
    parser.add_argument("--dataset", type=str, default="davis", choices=DATASET_CHOICES)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "results" / "best_model.pth")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-force-download", action="store_true")
    parser.add_argument("--prepare-force-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.results_dir)
    processed_csv = prepare_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        force_download=args.prepare_force_download,
        force_preprocess=args.prepare_force_preprocess,
        seed=args.seed,
    )
    splits = build_split_datasets(processed_csv_path=processed_csv, max_protein_len=1000)
    test_loader = DataLoader(
        splits.test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dta_batch,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.model_path, device=device)

    y_true, y_pred, test_loss = test_model(model, test_loader, device)
    metrics = regression_metrics(y_true, y_pred)
    save_scatter_plot(
        y_true,
        y_pred,
        args.results_dir / f"{args.dataset}_scatter_plot.png",
        title=f"{args.dataset} Predicted vs Actual",
    )

    report_text = (
        f"Evaluation on {args.dataset}:\n"
        f"[Testing] Loss: {test_loss:.6f}\n"
        f"[Metrics]\n"
        f"  - Concordance Index (CI): {metrics['ci']:.6f}\n"
        f"  - Mean Squared Error (MSE): {metrics['mse']:.6f}\n"
        f"  - Mean Absolute Error (MAE): {metrics['mae']:.6f}\n"
        f"  - Pearson Correlation (R): {metrics['pearson_r']:.6f}\n"
        f"  - RMSE: {metrics['rmse']:.6f}"
    )
    print(report_text)
    print("--------------------------------------------------")

    out_file = args.results_dir / f"{args.dataset}_performance.txt"
    out_file.write_text(report_text + "\n", encoding="utf-8")
    print(f"Evaluation complete. Results saved to ./results/{args.dataset}_performance.txt")


if __name__ == "__main__":
    main()
