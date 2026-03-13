from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from gcdta.data.featurizers import atom_feature_dim, bond_feature_dim, protein_physchem_dim, protein_vocab_size
from gcdta.models.gcdta import GCDTA


def build_model(config: Dict[str, float | int]) -> GCDTA:
    ablations = config.get("ablations", {})
    return GCDTA(
        atom_in_dim=int(config.get("atom_in_dim", atom_feature_dim())),
        bond_in_dim=int(config.get("bond_in_dim", bond_feature_dim())),
        protein_vocab_size=int(config.get("protein_vocab_size", protein_vocab_size())),
        target_physchem_dim=int(config.get("target_physchem_dim", protein_physchem_dim())),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        temperature=float(config.get("temperature", 0.2)),
        edge_dropout=float(config.get("edge_dropout", 0.2)),
        feature_dropout=float(config.get("feature_dropout", 0.1)),
        use_cross_attention=not bool(ablations.get("disable_cross_attention", False)),
        use_physchem_features=not bool(ablations.get("disable_physchem", False)),
        use_contrastive=not bool(ablations.get("disable_contrastive", False)),
        store_attention=bool(config.get("save_attention_matrix", False)),
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
    state_dict = checkpoint["model_state_dict"]
    
    # Handle wrapped models (EnhancedGCDTA saves with "base_model." prefix)
    # Strip the prefix to load into plain GCDTA
    if any(k.startswith("base_model.") for k in state_dict.keys()):
        state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items() 
                     if k.startswith("base_model.")}
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {path} is incompatible with the current model definition. "
            "Retrain the model to produce a new checkpoint."
        ) from exc
    model.to(device)
    return model, config
