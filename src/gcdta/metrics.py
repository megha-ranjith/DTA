from __future__ import annotations

import math
from typing import Dict

import numpy as np


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, chunk_size: int = 512) -> float:
    """Compute CI with chunking to avoid allocating full O(n^2) matrices."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = y_true.shape[0]
    if n < 2:
        return 0.0

    concordant = 0.0
    comparable = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yt = y_true[start:end][:, None]
        yp = y_pred[start:end][:, None]

        diff_true = yt - y_true[None, :]
        valid = diff_true > 0
        comparable += float(valid.sum())

        diff_pred = yp - y_pred[None, :]
        concordant += float(((diff_pred > 0) & valid).sum())
        concordant += 0.5 * float(((diff_pred == 0) & valid).sum())

    if comparable == 0:
        return 0.0
    return concordant / comparable


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mse(y_true, y_pred)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return 0.0
    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)
    if y_true_std == 0 or y_pred_std == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "ci": concordance_index(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }

