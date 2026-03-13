from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE with diagonal positives and in-batch negatives."""
    if z1.numel() == 0 or z2.numel() == 0:
        return z1.new_tensor(0.0)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = (z1 @ z2.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)
