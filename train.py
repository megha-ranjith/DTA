from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gcdta.data.dataset import build_split_datasets, collate_dta_batch
from gcdta.data.featurizers import atom_feature_dim, bond_feature_dim, protein_physchem_dim, protein_vocab_size
from gcdta.data.preprocess import prepare_dataset
from gcdta.metrics import regression_metrics
from gcdta.runtime import build_model, load_checkpoint, save_checkpoint
from gcdta.train_utils import (
    ensure_dir,
    grad_norm,
    save_logs_csv,
    save_scatter_plot,
    save_training_curves,
    set_seed,
    to_device,
)


DATASET_CHOICES = ["davis", "kiba", "core2016", "test71", "test105", "pdbbind_v2016"]


def _summarize_predictions(pred_sum: float, pred_sq_sum: float, count: int) -> Tuple[float, float]:
    if count == 0:
        return 0.0, 0.0
    mean = pred_sum / count
    variance = max(pred_sq_sum / count - mean * mean, 0.0)
    return mean, float(np.sqrt(variance))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    contrastive_weight: float,
) -> Dict[str, float]:
    model.train()
    criterion = nn.MSELoss()
    running_total = 0.0
    running_mse = 0.0
    running_cl = 0.0
    running_grad = 0.0
    pred_sum = 0.0
    pred_sq_sum = 0.0
    count = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = to_device(batch, device)
        optimizer.zero_grad()

        pred, cl_loss = model(batch)
        mse_loss = criterion(pred, batch["affinity"])
        loss = mse_loss + contrastive_weight * cl_loss
        loss.backward()
        running_grad += grad_norm(model.parameters())
        # Gradient clipping with max_norm=5.0 to stabilize training and prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = int(batch["affinity"].shape[0])
        running_total += float(loss.item()) * bs
        running_mse += float(mse_loss.item()) * bs
        running_cl += float(cl_loss.item()) * bs
        pred_sum += float(pred.detach().sum().item())
        pred_sq_sum += float((pred.detach() ** 2).sum().item())
        count += bs

    pred_mean, pred_std = _summarize_predictions(pred_sum, pred_sq_sum, count)
    return {
        "loss": running_total / max(count, 1),
        "mse": running_mse / max(count, 1),
        "contrastive": running_cl / max(count, 1),
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "grad_norm": running_grad / max(len(loader), 1),
    }


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0
    pred_sum = 0.0
    pred_sq_sum = 0.0
    count = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = to_device(batch, device)
        pred, _ = model(batch)
        loss = criterion(pred, batch["affinity"])
        bs = int(batch["affinity"].shape[0])
        running_loss += float(loss.item()) * bs
        pred_sum += float(pred.sum().item())
        pred_sq_sum += float((pred ** 2).sum().item())
        count += bs

    pred_mean, pred_std = _summarize_predictions(pred_sum, pred_sq_sum, count)
    return {
        "loss": running_loss / max(count, 1),
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for batch in tqdm(loader, desc="Testing", leave=False):
        batch = to_device(batch, device)
        pred, _ = model(batch)
        all_true.append(batch["affinity"].detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GCDTA for drug-target affinity prediction")
    parser.add_argument("--dataset", type=str, default="davis", choices=DATASET_CHOICES)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--edge-dropout", type=float, default=0.2)
    parser.add_argument("--feature-dropout", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--max-protein-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
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

    splits = build_split_datasets(processed_csv_path=processed_csv, max_protein_len=args.max_protein_len)
    train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_dta_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dta_batch,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dta_batch,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config: Dict[str, float | int] = {
        "atom_in_dim": atom_feature_dim(),
        "bond_in_dim": bond_feature_dim(),
        "protein_vocab_size": protein_vocab_size(),
        "target_physchem_dim": protein_physchem_dim(),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "temperature": float(args.temperature),
        "edge_dropout": float(args.edge_dropout),
        "feature_dropout": float(args.feature_dropout),
        "max_protein_len": int(args.max_protein_len),
    }
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    best_val = float("inf")
    best_epoch = 0
    best_model_path = args.results_dir / "best_model.pth"
    logs: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            contrastive_weight=args.contrastive_weight,
        )
        val_stats = evaluate_loss(model=model, loader=val_loader, device=device)
        scheduler.step(val_stats["loss"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_mse": train_stats["mse"],
                "train_contrastive": train_stats["contrastive"],
                "train_pred_mean": train_stats["pred_mean"],
                "train_pred_std": train_stats["pred_std"],
                "grad_norm": train_stats["grad_norm"],
                "val_loss": val_stats["loss"],
                "val_pred_mean": val_stats["pred_mean"],
                "val_pred_std": val_stats["pred_std"],
                "lr": current_lr,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_stats['loss']:.6f} | val_loss={val_stats['loss']:.6f} "
            f"| pred_mean={val_stats['pred_mean']:.4f} | pred_std={val_stats['pred_std']:.4f} "
            f"| grad_norm={train_stats['grad_norm']:.4f} | lr={current_lr:.2e}"
        )
        if val_stats["pred_std"] < 0.05:
            print("Warning: validation prediction std is near zero. The model may be collapsing to the mean.")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_epoch = epoch
            save_checkpoint(model=model, config=config, path=best_model_path)

    save_logs_csv(logs, args.results_dir / "logs.csv")
    save_training_curves(logs, args.results_dir / "training_curves.png")

    best_model, _ = load_checkpoint(best_model_path, device=device)
    y_true, y_pred = predict(best_model, test_loader, device=device)
    metrics = regression_metrics(y_true, y_pred)
    save_scatter_plot(y_true, y_pred, args.results_dir / "scatter_plot.png", title=f"{args.dataset} Predicted vs Actual")

    summary = {
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_metrics": metrics,
        "config": {**config, "lr": args.lr, "batch_size": args.batch_size, "weight_decay": args.weight_decay},
    }
    (args.results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Logs saved to: {args.results_dir / 'logs.csv'}")
    print(f"Training curves saved to: {args.results_dir / 'training_curves.png'}")
    print(f"Scatter plot saved to: {args.results_dir / 'scatter_plot.png'}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
