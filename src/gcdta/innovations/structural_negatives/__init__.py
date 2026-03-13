"""
Innovation 4: Structural Hard Negatives (Decoy Training)

Use structurally similar compounds (decoys) as hard negatives
instead of random negatives to improve contrastive learning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoySampler(nn.Module):
    """
    Sample hard negatives (decoys) for contrastive learning.

    Decoys are structurally similar to actives but with different binding affinity.
    """

    def __init__(self, decoy_ratio: float = 1.0, use_structural_similarity: bool = True):
        super().__init__()
        self.decoy_ratio = decoy_ratio
        self.use_structural_similarity = use_structural_similarity

    def sample_decoys(
        self,
        positive_smiles: List[str],
        decoy_pool: List[str],
        similarity_scores: torch.Tensor,  # (batch_size, pool_size)
        affinity_difference_threshold: float = 1.0,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Select hard negatives from decoy pool.

        Args:
            positive_smiles: Original positive samples
            decoy_pool: Available decoys
            similarity_scores: Similarity between positives and decoys
            affinity_difference_threshold: Minimum affinity difference for hard negative

        Returns:
            selected_decoys: Selected hard negative SMILES
            hard_negative_mask: Which decoys are hard negatives
        """
        batch_size = len(positive_smiles)
        num_decoys = len(decoy_pool)

        # Find structurally similar compounds (high similarity)
        hard_negative_similarity = torch.topk(similarity_scores, k=int(
            num_decoys * self.decoy_ratio
        ), dim=1).values

        # In real scenario, filter by affinity difference
        # For now, use top-k structurally similar

        selected_decoys = []
        hard_negative_mask = torch.zeros(batch_size, num_decoys, dtype=torch.bool)

        for i in range(batch_size):
            top_indices = torch.topk(similarity_scores[i], k=max(1, int(num_decoys * self.decoy_ratio))).indices
            for idx in top_indices:
                if idx < len(decoy_pool):
                    if i == 0 or len(selected_decoys) <= batch_size:
                        selected_decoys.append(decoy_pool[idx.item()])
                        hard_negative_mask[i, idx] = True

        return selected_decoys, hard_negative_mask


class StructuralPerturbation(nn.Module):
    """
    Generate structural perturbations (augmentations) of molecules.

    Methods:
    - Conformational perturbation (3D coordinate noise)
    - Pocket coordinate perturbation
    """

    def __init__(self):
        super().__init__()

    def perturb_conformation(
        self,
        graph_features: torch.Tensor,  # (batch, n_atoms, F)
        perturbation_std: float = 0.1,
    ) -> torch.Tensor:
        """
        Add Gaussian noise to 3D coordinates.

        Args:
            graph_features: Node features and coordinates
            perturbation_std: Standard deviation of noise

        Returns:
            perturbed_features: Noisy features
        """
        # Add small Gaussian noise to coordinates (assuming last 3 dims are coords)
        noise = torch.randn_like(graph_features) * perturbation_std
        perturbed = graph_features + noise
        return perturbed

    def perturb_pocket(
        self,
        protein_coords: torch.Tensor,  # (batch, n_residues, 3)
        perturbation_std: float = 0.2,
    ) -> torch.Tensor:
        """
        Perturb pocket coordinates.

        Args:
            protein_coords: Protein residue coordinates
            perturbation_std: Standard deviation of noise

        Returns:
            perturbed_coords: Noisy coordinates
        """
        noise = torch.randn_like(protein_coords) * perturbation_std
        perturbed = protein_coords + noise
        return perturbed

    def compute_rmsd(
        self,
        coords1: torch.Tensor,  # (N, 3)
        coords2: torch.Tensor,  # (N, 3)
    ) -> torch.Tensor:
        """
        Compute RMSD between two coordinate sets.

        Args:
            coords1: Original coordinates
            coords2: Perturbed coordinates

        Returns:
            rmsd: Root mean squared deviation
        """
        diff = coords1 - coords2
        mse = (diff ** 2).sum(dim=-1).mean()
        rmsd = torch.sqrt(mse + 1e-8)
        return rmsd


class StructuralContrastiveLoss(nn.Module):
    """
    Contrastive loss with structural hard negatives.

    Loss = -log(exp(sim(pos)) / (exp(sim(pos)) + sum_i(w_i * exp(sim(neg_i)))))

    Where w_i are weights based on structural similarity (RMSD).
    """

    def __init__(self, temperature: float = 0.2, use_rmsd_weight: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_rmsd_weight = use_rmsd_weight

    def forward(
        self,
        anchor_embedding: torch.Tensor,  # (batch, dim)
        positive_embedding: torch.Tensor,  # (batch, dim)
        negative_embeddings: torch.Tensor,  # (batch, num_negatives, dim)
        negative_rmsds: torch.Tensor = None,  # (batch, num_negatives) - optional
    ) -> torch.Tensor:
        """
        Compute structural contrastive loss.

        Args:
            anchor_embedding: Anchor embeddings
            positive_embedding: Positive sample embeddings
            negative_embeddings: Hard negative embeddings
            negative_rmsds: RMSD values for weighting (optional)

        Returns:
            loss: Scalar loss value
        """
        batch_size = anchor_embedding.shape[0]

        # Normalize embeddings
        anchor = F.normalize(anchor_embedding, p=2, dim=-1)  # (batch, dim)
        positive = F.normalize(positive_embedding, p=2, dim=-1)  # (batch, dim)
        negatives = F.normalize(negative_embeddings, p=2, dim=-1)  # (batch, num_neg, dim)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # (batch,)

        # Negative similarities
        neg_sims = torch.einsum("bd,bnd->bn", anchor, negatives) / self.temperature  # (batch, num_neg)

        # Compute weights based on RMSD if available
        if self.use_rmsd_weight and negative_rmsds is not None:
            # Higher RMSD = harder negative = higher weight
            weights = torch.clamp(negative_rmsds / 2.0, 0.1, 1.0)  # (batch, num_neg)
            weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize
        else:
            weights = torch.ones_like(neg_sims) / neg_sims.shape[1]

        # Weighted negative similarities
        weighted_neg_sims = (weights * neg_sims).sum(dim=1)  # (batch,)

        # Contrastive loss
        loss = -pos_sim + torch.logsumexp(
            torch.stack([pos_sim, weighted_neg_sims], dim=1), dim=1
        )

        return loss.mean()


class StructuralNegativesModule(nn.Module):
    """
    Structural hard negative mining with contrastive loss.
    
    Computes InfoNCE loss for drug-target-decoy triplets.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        temperature: float = 0.2,
        use_rmsd_weight: bool = True,
        decoy_ratio: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.use_rmsd_weight = use_rmsd_weight
        
        # Projection layer for decoy embeddings if needed
        self.decoy_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.sampler = DecoySampler(decoy_ratio=decoy_ratio)
        self.perturbation = StructuralPerturbation()

    def forward(
        self,
        anchor_embeddings: torch.Tensor = None,
        positive_embeddings: torch.Tensor = None,
        negative_embeddings: torch.Tensor = None,
        negative_rmsds: torch.Tensor = None,
        drug_embedding: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
        decoy_embeddings: torch.Tensor = None,
        affinity: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute structural contrastive loss.

        Args:
            anchor_embeddings: Drug embeddings (batch, dim)
            positive_embeddings: Target embeddings (batch, dim)
            negative_embeddings: Decoy embeddings (batch, num_neg, dim) or (batch, dim)
            negative_rmsds: RMSD values for weighting (batch, num_neg)
            
            OR alternative names:
            drug_embedding: Same as anchor_embeddings
            target_embedding: Same as positive_embeddings
            decoy_embeddings: Same as negative_embeddings

        Returns:
            loss: Contrastive loss (scalar)
            pos_similarity: Positive pair similarity (scalar)
            neg_similarity: Negative pair similarity (scalar)
        """
        # Handle alternative parameter names
        if anchor_embeddings is None and drug_embedding is not None:
            anchor_embeddings = drug_embedding
        if positive_embeddings is None and target_embedding is not None:
            positive_embeddings = target_embedding
        if negative_embeddings is None and decoy_embeddings is not None:
            negative_embeddings = decoy_embeddings
        
        # Validate inputs
        if anchor_embeddings is None or positive_embeddings is None:
            raise ValueError("anchor_embeddings and positive_embeddings are required")
        
        batch_size = anchor_embeddings.shape[0]
        
        # Pool embeddings if 3D
        if anchor_embeddings.dim() == 3:
            anchor_embeddings = anchor_embeddings.mean(dim=1)
        if positive_embeddings.dim() == 3:
            positive_embeddings = positive_embeddings.mean(dim=1)
        
        # Normalize embeddings
        anchor = F.normalize(anchor_embeddings, p=2, dim=-1)  # (batch, dim)
        positive = F.normalize(positive_embeddings, p=2, dim=-1)  # (batch, dim)
        
        # Positive similarity: cosine similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # (batch,)
        
        # Handle negative embeddings
        if negative_embeddings is not None:
            # Handle 3D negatives (batch, num_neg, dim)
            if negative_embeddings.dim() == 3:
                negatives = F.normalize(negative_embeddings, p=2, dim=-1)  # (batch, num_neg, dim)
                # Compute similarities
                neg_sims = torch.einsum("bd,bnd->bn", anchor, negatives) / self.temperature  # (batch, num_neg)
            # Handle 2D negatives (batch, dim)
            elif negative_embeddings.dim() == 2:
                negatives = F.normalize(negative_embeddings, p=2, dim=-1)  # (batch, dim)
                neg_sims = (anchor * negatives).sum(dim=-1, keepdim=True) / self.temperature  # (batch, 1)
            else:
                raise ValueError(f"Invalid negative_embeddings shape: {negative_embeddings.shape}")
            
            # Apply RMSD weighting if available
            if negative_rmsds is not None and negative_rmsds.numel() > 0:
                # Ensure RMSD has correct shape
                if negative_rmsds.dim() == 1:
                    negative_rmsds = negative_rmsds.unsqueeze(-1)  # (batch,) -> (batch, 1)
                
                # Weight by RMSD: higher RMSD = harder negative = higher weight
                weights = torch.clamp(negative_rmsds / 2.0, 0.1, 1.0)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                weights = torch.ones_like(neg_sims) / (neg_sims.shape[-1] + 1e-8)
            
            # Weighted negative similarity
            weighted_neg_sims = (weights * neg_sims).sum(dim=1)  # (batch,)
        else:
            # If no negatives provided, use small negative value
            weighted_neg_sims = -torch.ones(batch_size, device=anchor.device) * 10.0
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + exp(neg)))
        # Using logsumexp for numerical stability
        logits = torch.stack([pos_sim, weighted_neg_sims], dim=1)  # (batch, 2)
        loss = -pos_sim + torch.logsumexp(logits, dim=1)  # (batch,)
        loss_val = loss.mean()
        
        # Compute similarities for monitoring
        pos_similarity = torch.sigmoid(pos_sim).mean()
        neg_similarity = torch.sigmoid(weighted_neg_sims).mean()
        
        if return_dict:
            outputs: Dict[str, torch.Tensor] = {
                "contrastive_loss": loss_val,
                "positive_similarity": pos_similarity,
                "negative_similarity": neg_similarity,
            }
            if affinity is not None:
                outputs["affinity"] = affinity
            return outputs

        return loss_val, pos_similarity, neg_similarity
