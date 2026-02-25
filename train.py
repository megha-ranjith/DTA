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
from gcdta.data.featurizers import atom_feature_dim
from gcdta.data.preprocess import prepare_dataset
from gcdta.metrics import regression_metrics
from gcdta.runtime import build_model, load_checkpoint, save_checkpoint
from gcdta.train_utils import ensure_dir, save_logs_csv, save_scatter_plot, to_device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    contrastive_weight: float,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = to_device(batch, device)
        optimizer.zero_grad()

        pred, cl_loss = model(batch)
        mse_loss = criterion(pred, batch["affinity"])
        loss = mse_loss + contrastive_weight * cl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = int(batch["affinity"].shape[0])
        running_loss += float(loss.item()) * bs
        count += bs

    return running_loss / max(count, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = to_device(batch, device)
        pred, _ = model(batch)
        loss = criterion(pred, batch["affinity"])
        bs = int(batch["affinity"].shape[0])
        running_loss += float(loss.item()) * bs
        count += bs

    return running_loss / max(count, 1)


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
    parser.add_argument("--dataset", type=str, default="davis", choices=["davis", "kiba", "pdbbind_v2016"])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--edge-dropout", type=float, default=0.2)
    parser.add_argument("--feature-dropout", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--max-protein-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prepare-force-download", action="store_true")
    parser.add_argument("--prepare-force-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        "target_in_dim": 40,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "temperature": float(args.temperature),
        "edge_dropout": float(args.edge_dropout),
        "feature_dropout": float(args.feature_dropout),
        "max_protein_len": int(args.max_protein_len),
    }
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_model_path = args.results_dir / "best_model.pth"
    logs: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            contrastive_weight=args.contrastive_weight,
        )
        val_loss = evaluate_loss(model=model, loader=val_loader, device=device)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model=model, config=config, path=best_model_path)

    save_logs_csv(logs, args.results_dir / "logs.csv")

    best_model, _ = load_checkpoint(best_model_path, device=device)
    y_true, y_pred = predict(best_model, test_loader, device=device)
    metrics = regression_metrics(y_true, y_pred)
    save_scatter_plot(y_true, y_pred, args.results_dir / "scatter_plot.png")

    summary = {
        "dataset": args.dataset,
        "best_val_loss": best_val,
        "test_metrics": metrics,
    }
    (args.results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Logs saved to: {args.results_dir / 'logs.csv'}")
    print(f"Scatter plot saved to: {args.results_dir / 'scatter_plot.png'}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()

