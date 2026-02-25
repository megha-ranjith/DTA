from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from gcdta.losses import info_nce_loss


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    num = (values * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1e-8)
    return num / den


class DrugGATEncoder(nn.Module):
    """SMILES molecular graph encoder with 8-head graph attention."""

    def __init__(self, atom_in_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.gat1 = GATConv(atom_in_dim, hidden_dim, heads=8, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * 8, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch_idx = graph_batch.batch

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        tokens, token_mask = to_dense_batch(x, batch_idx)
        graph_emb = global_mean_pool(x, batch_idx)
        return tokens, token_mask, graph_emb


class DilatedTargetEncoder(nn.Module):
    """Multi-layer dilated CNN target encoder with PReLU activation."""

    def __init__(
        self,
        target_in_dim: int,
        hidden_dim: int = 128,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(target_in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.activations = nn.ModuleList([nn.PReLU() for _ in dilations])
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_feat: torch.Tensor, target_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(target_feat)  # [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]

        for conv, activation in zip(self.convs, self.activations):
            x = conv(x)
            x = activation(x)
            x = self.dropout(x)

        x = x.transpose(1, 2)  # [B, L, D]
        x = x * target_mask.unsqueeze(-1)
        graph_emb = masked_mean(x, target_mask, dim=1)
        return x, graph_emb


class CrossAttentionFusion(nn.Module):
    """Cross-attention module to fuse drug and protein embeddings."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.drug_to_target = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.target_to_drug = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_drug = nn.LayerNorm(hidden_dim)
        self.norm_target = nn.LayerNorm(hidden_dim)
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
        drug_ctx, _ = self.drug_to_target(
            query=drug_tokens,
            key=target_tokens,
            value=target_tokens,
            key_padding_mask=~target_mask,
        )
        drug_tokens = self.norm_drug(drug_tokens + self.dropout(drug_ctx))

        target_ctx, _ = self.target_to_drug(
            query=target_tokens,
            key=drug_tokens,
            value=drug_tokens,
            key_padding_mask=~drug_mask,
        )
        target_tokens = self.norm_target(target_tokens + self.dropout(target_ctx))

        drug_pool = masked_mean(drug_tokens, drug_mask, dim=1)
        target_pool = masked_mean(target_tokens, target_mask, dim=1)
        return torch.cat([drug_pool, target_pool, drug_graph, target_graph], dim=-1)


class HGCNContrastive(nn.Module):
    """HGCN branch for contrastive representation learning with InfoNCE."""

    def __init__(
        self,
        hidden_dim: int = 128,
        temperature: float = 0.2,
        edge_dropout: float = 0.2,
        feature_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.edge_dropout = edge_dropout
        self.feature_dropout = feature_dropout
        self.temperature = temperature

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _aggregate_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        out = values.new_zeros((size, values.size(-1)))
        cnt = values.new_zeros((size, 1))
        out.index_add_(0, index, values)
        ones = values.new_ones((values.size(0), 1))
        cnt.index_add_(0, index, ones)
        return out / cnt.clamp_min(1.0)

    @staticmethod
    def _drop_edges(edge_index: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
        if (not training) or drop_prob <= 0.0 or edge_index.numel() == 0:
            return edge_index
        num_edges = edge_index.size(1)
        keep = torch.rand(num_edges, device=edge_index.device) > drop_prob
        if keep.sum() == 0:
            keep[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True
        return edge_index[:, keep]

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.gcn2(h, edge_index)
        return h

    def forward(
        self,
        drug_emb: torch.Tensor,
        target_emb: torch.Tensor,
        drug_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_drug_ids, drug_local = torch.unique(drug_ids, sorted=True, return_inverse=True)
        unique_target_ids, target_local = torch.unique(target_ids, sorted=True, return_inverse=True)

        num_drugs = int(unique_drug_ids.numel())
        num_targets = int(unique_target_ids.numel())

        drug_nodes = self._aggregate_mean(drug_emb, drug_local, num_drugs)
        target_nodes = self._aggregate_mean(target_emb, target_local, num_targets)
        x = torch.cat([drug_nodes, target_nodes], dim=0)

        src = drug_local
        dst = target_local + num_drugs
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        base = self._encode(x, edge_index)
        sample_drug_ctx = base[drug_local]
        sample_target_ctx = base[target_local + num_drugs]

        if not self.training:
            return x.new_tensor(0.0), sample_drug_ctx, sample_target_ctx

        edge_index_1 = self._drop_edges(edge_index, self.edge_dropout, self.training)
        edge_index_2 = self._drop_edges(edge_index, self.edge_dropout, self.training)
        x1 = F.dropout(x, p=self.feature_dropout, training=self.training)
        x2 = F.dropout(x, p=self.feature_dropout, training=self.training)

        z1 = self.projector(self._encode(x1, edge_index_1))
        z2 = self.projector(self._encode(x2, edge_index_2))
        cl_loss = info_nce_loss(z1, z2, temperature=self.temperature)
        return cl_loss, sample_drug_ctx, sample_target_ctx

