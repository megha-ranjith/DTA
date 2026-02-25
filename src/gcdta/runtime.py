from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from gcdta.data.featurizers import atom_feature_dim
from gcdta.models.gcdta import GCDTA


def build_model(config: Dict[str, float | int]) -> GCDTA:
    return GCDTA(
        atom_in_dim=int(config.get("atom_in_dim", atom_feature_dim())),
        target_in_dim=int(config.get("target_in_dim", 40)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        temperature=float(config.get("temperature", 0.2)),
        edge_dropout=float(config.get("edge_dropout", 0.2)),
        feature_dropout=float(config.get("feature_dropout", 0.1)),
    )


def save_checkpoint(model: GCDTA, config: Dict[str, float | int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": dict(config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> Tuple[GCDTA, Dict[str, float | int]]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, config

