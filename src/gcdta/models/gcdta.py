from __future__ import annotations

import torch
import torch.nn as nn

from .components import CrossAttentionFusion, DilatedTargetEncoder, DrugGATEncoder, HGCNContrastive


class GCDTA(nn.Module):
    def __init__(
        self,
        atom_in_dim: int,
        target_in_dim: int = 40,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        temperature: float = 0.2,
        edge_dropout: float = 0.2,
        feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = DrugGATEncoder(atom_in_dim=atom_in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.target_encoder = DilatedTargetEncoder(
            target_in_dim=target_in_dim,
            hidden_dim=hidden_dim,
            dilations=(1, 2, 4, 8),
            dropout=dropout,
        )
        self.fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=8, dropout=dropout)
        self.hgcn_contrastive = HGCNContrastive(
            hidden_dim=hidden_dim,
            temperature=temperature,
            edge_dropout=edge_dropout,
            feature_dropout=feature_dropout,
        )

        fusion_dim = hidden_dim * 4
        regression_in = fusion_dim + hidden_dim * 2
        self.regressor = nn.Sequential(
            nn.Linear(regression_in, 512),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, batch):
        drug_tokens, drug_mask, drug_graph = self.drug_encoder(batch["drug_graph"])
        target_tokens, target_graph = self.target_encoder(batch["target_feat"], batch["target_mask"])
        target_mask = batch["target_mask"] > 0

        fused = self.fusion(
            drug_tokens=drug_tokens,
            drug_mask=drug_mask,
            target_tokens=target_tokens,
            target_mask=target_mask,
            drug_graph=drug_graph,
            target_graph=target_graph,
        )

        cl_loss, drug_ctx, target_ctx = self.hgcn_contrastive(
            drug_emb=drug_graph,
            target_emb=target_graph,
            drug_ids=batch["drug_node_id"],
            target_ids=batch["target_node_id"],
        )

        pred_input = torch.cat([fused, drug_ctx, target_ctx], dim=-1)
        pred = self.regressor(pred_input).squeeze(-1)
        return pred, cl_loss

