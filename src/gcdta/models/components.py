from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
from torch_geometric.utils import dropout_edge, to_dense_batch

from gcdta.losses import info_nce_loss


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    numerator = (values * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(1e-8)
    return numerator / denominator


class DrugGATEncoder(nn.Module):
    """Edge-aware GAT encoder over molecular graphs."""

    def __init__(
        self,
        atom_in_dim: int,
        bond_in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(atom_in_dim, hidden_dim)
        self.gat1 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=8,
            concat=False,
            edge_dim=bond_in_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.gat2 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=8,
            concat=False,
            edge_dim=bond_in_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(graph_batch.x)
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch_idx = graph_batch.batch

        h = self.gat1(x, edge_index, edge_attr)
        h = self.norm1(x + self.dropout(F.elu(h)))
        h2 = self.gat2(h, edge_index, edge_attr)
        h = self.norm2(h + self.dropout(F.elu(h2)))

        tokens, token_mask = to_dense_batch(h, batch_idx)
        graph_emb = global_mean_pool(h, batch_idx)
        return tokens, token_mask, graph_emb


class ResidualDilatedBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Transpose back to (batch, seq_len, hidden_dim) for LayerNorm
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        return x + out


class DilatedTargetEncoder(nn.Module):
    """Protein encoder with learned residue embeddings and dilated CNN blocks."""

    def __init__(
        self,
        vocab_size: int,
        physchem_dim: int,
        hidden_dim: int = 128,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
        use_physchem_features: bool = True,
    ) -> None:
        super().__init__()
        residue_dim = hidden_dim // 2
        self.embedding = nn.Embedding(vocab_size, residue_dim)
        self.input_proj = nn.Linear(residue_dim + physchem_dim, hidden_dim)
        self.use_physchem_features = use_physchem_features
        self.blocks = nn.ModuleList(
            [ResidualDilatedBlock(hidden_dim=hidden_dim, dilation=dilation, dropout=dropout) for dilation in dilations]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        physchem: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residue_emb = self.embedding(token_ids)
        if not self.use_physchem_features:
            physchem = torch.zeros_like(physchem)
        x = torch.cat([residue_emb, physchem], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.output_norm(x)
        x = x * target_mask.unsqueeze(-1)
        graph_emb = masked_mean(x, target_mask, dim=1)
        return x, graph_emb


class CrossAttentionFusion(nn.Module):
    """Cross-attention with drug atoms as query and protein residues as key/value."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_cross_attention: bool = True,
        store_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.store_attention = store_attention
        self.last_attention_weights: torch.Tensor | None = None
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_tokens: torch.Tensor,
        drug_mask: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        drug_graph: torch.Tensor,
        target_graph: torch.Tensor,
    ) -> torch.Tensor:
        # key_padding_mask should be True for positions to mask (padding), False for valid
        # target_mask is True for valid positions, so invert to get padding mask
        target_padding_mask = ~target_mask if target_mask.dtype == torch.bool else (target_mask == 0)

        self.last_attention_weights = None
        if self.use_cross_attention:
            attended, attention_weights = self.cross_attention(
                query=drug_tokens,
                key=target_tokens,
                value=target_tokens,
                key_padding_mask=target_padding_mask,
                need_weights=self.store_attention,
                average_attn_weights=True,
            )
            if attention_weights is not None:
                self.last_attention_weights = attention_weights.detach().cpu()
            drug_tokens = self.norm1(drug_tokens + self.dropout(attended))
            drug_tokens = self.norm2(drug_tokens + self.dropout(self.feed_forward(drug_tokens)))

        drug_pool = masked_mean(drug_tokens, drug_mask, dim=1)
        target_pool = masked_mean(target_tokens, target_mask, dim=1)
        return torch.cat([drug_pool, target_pool, drug_graph, target_graph], dim=-1)


class HGCNContrastive(nn.Module):
    """Batch-level heterogeneous graph encoder with pair-wise InfoNCE."""

    def __init__(
        self,
        hidden_dim: int = 128,
        temperature: float = 0.2,
        edge_dropout: float = 0.2,
        feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.edge_dropout = edge_dropout
        self.feature_dropout = feature_dropout
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.projector_drug = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projector_target = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _aggregate_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        out = values.new_zeros((size, values.size(-1)))
        count = values.new_zeros((size, 1))
        out.index_add_(0, index, values)
        count.index_add_(0, index, values.new_ones((values.size(0), 1)))
        return out / count.clamp_min(1.0)

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.feature_dropout, training=self.training)
        h = self.gcn2(h, edge_index)
        return h

    def forward(
        self,
        drug_emb: torch.Tensor,
        target_emb: torch.Tensor,
        drug_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, drug_local = torch.unique(drug_ids, sorted=True, return_inverse=True)
        _, target_local = torch.unique(target_ids, sorted=True, return_inverse=True)
        num_drugs = int(drug_local.max().item()) + 1
        num_targets = int(target_local.max().item()) + 1

        drug_nodes = self._aggregate_mean(drug_emb, drug_local, num_drugs)
        target_nodes = self._aggregate_mean(target_emb, target_local, num_targets)
        node_features = torch.cat([drug_nodes, target_nodes], dim=0)

        src = drug_local
        dst = target_local + num_drugs
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        # Generate two augmented views for contrastive learning
        if self.training and self.edge_dropout > 0.0:
            # First augmented view
            edge_index_1, _ = dropout_edge(edge_index, p=self.edge_dropout, training=True)
            if edge_index_1.numel() == 0:
                edge_index_1 = edge_index
            
            # Second augmented view
            edge_index_2, _ = dropout_edge(edge_index, p=self.edge_dropout, training=True)
            if edge_index_2.numel() == 0:
                edge_index_2 = edge_index
        else:
            edge_index_1 = edge_index
            edge_index_2 = edge_index

        # Encode both augmented views
        encoded_1 = self._encode(node_features, edge_index_1)
        encoded_2 = self._encode(node_features, edge_index_2)

        sample_drug_ctx_1 = encoded_1[drug_local]
        sample_target_ctx_1 = encoded_1[target_local + num_drugs]
        sample_drug_ctx_2 = encoded_2[drug_local]
        sample_target_ctx_2 = encoded_2[target_local + num_drugs]

        if not self.training:
            # Return average context from both views
            sample_drug_ctx = (sample_drug_ctx_1 + sample_drug_ctx_2) / 2.0
            sample_target_ctx = (sample_target_ctx_1 + sample_target_ctx_2) / 2.0
            return encoded_1.new_tensor(0.0), sample_drug_ctx, sample_target_ctx

        # Project both views and compute contrastive loss
        proj_drug_1 = self.projector_drug(sample_drug_ctx_1)
        proj_drug_2 = self.projector_drug(sample_drug_ctx_2)
        proj_target_1 = self.projector_target(sample_target_ctx_1)
        proj_target_2 = self.projector_target(sample_target_ctx_2)

        # Compute contrastive loss between views
        cl_loss_drug = info_nce_loss(proj_drug_1, proj_drug_2, temperature=self.temperature)
        cl_loss_target = info_nce_loss(proj_target_1, proj_target_2, temperature=self.temperature)
        cl_loss = (cl_loss_drug + cl_loss_target) / 2.0

        # Return averaged context for predictions
        sample_drug_ctx = (sample_drug_ctx_1 + sample_drug_ctx_2) / 2.0
        sample_target_ctx = (sample_target_ctx_1 + sample_target_ctx_2) / 2.0
        return cl_loss, sample_drug_ctx, sample_target_ctx
