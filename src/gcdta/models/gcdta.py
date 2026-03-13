from __future__ import annotations

import torch
import torch.nn as nn

from .components import CrossAttentionFusion, DilatedTargetEncoder, DrugGATEncoder, HGCNContrastive


class GCDTA(nn.Module):
    def __init__(
        self,
        atom_in_dim: int,
        bond_in_dim: int,
        protein_vocab_size: int = 21,
        target_physchem_dim: int = 19,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        temperature: float = 0.2,
        edge_dropout: float = 0.2,
        feature_dropout: float = 0.1,
        use_cross_attention: bool = True,
        use_physchem_features: bool = True,
        use_contrastive: bool = True,
        store_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_contrastive = use_contrastive
        self.drug_encoder = DrugGATEncoder(
            atom_in_dim=atom_in_dim,
            bond_in_dim=bond_in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.target_encoder = DilatedTargetEncoder(
            vocab_size=protein_vocab_size,
            physchem_dim=target_physchem_dim,
            hidden_dim=hidden_dim,
            dilations=(1, 2, 4, 8),
            dropout=dropout,
            use_physchem_features=use_physchem_features,
        )
        self.fusion = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            use_cross_attention=use_cross_attention,
            store_attention=store_attention,
        )
        self.hgcn_contrastive = HGCNContrastive(
            hidden_dim=hidden_dim,
            temperature=temperature,
            edge_dropout=edge_dropout,
            feature_dropout=feature_dropout,
        )

        fusion_dim = hidden_dim * 4
        regression_in = fusion_dim + hidden_dim * 2  # Actual input: fused + drug_ctx + target_ctx
        
        # Simplified regression head with PReLU activation (unbounded, prevents collapse)
        # No activation on final layer to allow full range of predictions (4.5 - 11.0 pKd)
        self.regressor = nn.Sequential(
            nn.LayerNorm(regression_in),
            nn.Linear(regression_in, 256),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Initialize final regression layer bias to dataset mean (~6.8 for Davis)
        # Critical to prevent predictions from collapsing to near-zero at initialization
        with torch.no_grad():
            self.regressor[-1].bias.fill_(6.8)


    def forward(self, batch):
        drug_tokens, drug_mask, drug_graph = self.drug_encoder(batch["drug_graph"])
        target_tokens, target_graph = self.target_encoder(
            token_ids=batch["target_tokens"],
            physchem=batch["target_physchem"],
            target_mask=batch["target_mask"],
        )
        target_mask = batch["target_mask"] > 0

        fused = self.fusion(
            drug_tokens=drug_tokens,
            drug_mask=drug_mask,
            target_tokens=target_tokens,
            target_mask=target_mask,
            drug_graph=drug_graph,
            target_graph=target_graph,
        )

        if self.use_contrastive:
            cl_loss, drug_ctx, target_ctx = self.hgcn_contrastive(
                drug_emb=drug_graph,
                target_emb=target_graph,
                drug_ids=batch["drug_node_id"],
                target_ids=batch["target_node_id"],
            )
        else:
            cl_loss = drug_graph.new_tensor(0.0)
            drug_ctx = torch.zeros_like(drug_graph)
            target_ctx = torch.zeros_like(target_graph)

        pred_input = torch.cat([fused, drug_ctx, target_ctx], dim=-1)
        pred = self.regressor(pred_input).squeeze(-1)
        return pred, cl_loss
