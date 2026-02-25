from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_logs_csv(logs: List[Dict[str, float]], output_path: Path) -> None:
    df = pd.DataFrame(logs)
    df.to_csv(output_path, index=False)


def save_scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
    plt.xlabel("Actual Affinity")
    plt.ylabel("Predicted Affinity")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out

