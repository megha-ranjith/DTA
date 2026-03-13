from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_logs_csv(logs: List[Dict[str, float]], output_path: Path) -> None:
    df = pd.DataFrame(logs)
    df.to_csv(output_path, index=False)


def save_scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str = "Predicted vs Actual") -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("Actual Affinity (pKd)")
    plt.ylabel("Predicted Affinity (pKd)")
    plt.title(title)
    # Fix axes to standard affinity range (4.5 - 11.0 pKd) for consistent visualization
    plt.xlim(4.5, 11)
    plt.ylim(4.5, 11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_training_curves(logs: List[Dict[str, float]], output_path: Path) -> None:
    if not logs:
        return
    df = pd.DataFrame(logs)
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_prediction_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    dataset: str,
    split: str,
) -> None:
    df = pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "dataset": dataset,
            "split": split,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
        }
    )
    df.to_csv(output_path, index=False)


def save_attention_csv(
    attention: np.ndarray,
    row_labels: List[str],
    column_labels: List[str],
    output_path: Path,
) -> None:
    records = []
    for i, row_label in enumerate(row_labels):
        for j, column_label in enumerate(column_labels):
            records.append(
                {
                    "row_label": row_label,
                    "column_label": column_label,
                    "attention_value": float(attention[i, j]),
                }
            )
    pd.DataFrame(records).to_csv(output_path, index=False)


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2)
        total += float(param_norm.item() ** 2)
    return float(total ** 0.5)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out
