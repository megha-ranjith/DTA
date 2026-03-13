"""
Innovation 1: Uncertainty-Aware Pocket-Centric Modeling

Extracts binding pockets from 3D protein structures and performs
geometric learning with uncertainty estimation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

class PocketExtractor(nn.Module):
    """Extract binding pocket atoms from protein structure."""

    def __init__(self, pocket_radius_angstrom: float = 8.0):
        super().__init__()
        self.pocket_radius = pocket_radius_angstrom

    def extract_pocket(
        self,
        protein_coords: torch.Tensor,  # (N_residues, 3)
        ligand_coords: torch.Tensor,  # (M_atoms, 3)
        protein_features: torch.Tensor,  # (N_residues, F)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract pocket residues within radius of ligand center.

        Args:
            protein_coords: Protein backbone coordinates (N, 3)
            ligand_coords: Ligand atom coordinates (M, 3)
            protein_features: Protein residue features (N, F)

        Returns:
            pocket_coords: Residues within pocket (P, 3)
            pocket_features: Features of pocket residues (P, F)
            pocket_mask: Binary mask for original residues (N,)
        """
        ligand_center = ligand_coords.mean(dim=0, keepdim=True)  # (1, 3)
        distances = torch.norm(protein_coords - ligand_center, dim=1)  # (N,)
        pocket_mask = distances <= self.pocket_radius  # (N,)

        pocket_coords = protein_coords[pocket_mask]  # (P, 3)
        pocket_features = protein_features[pocket_mask]  # (P, F)

        return pocket_coords, pocket_features, pocket_mask

    def forward(
        self,
        protein_coords: torch.Tensor,
        ligand_coords: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.extract_pocket(protein_coords, ligand_coords, protein_features)


class GVPEncoder(nn.Module):
    """
    Geometric Vector Perceptron encoder for pocket.

    Learns from scalar and vector features of 3D structures.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Scalar feature projections
        self.scalar_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Vector feature processing (3D coordinates)
        self.vector_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.message_layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)]
        )

        # Edge encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # distance + angle features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        pocket_features: torch.Tensor,  # (P, F_scalar)
        pocket_coords: torch.Tensor,  # (P, 3)
    ) -> torch.Tensor:
        """
        Encode pocket geometry.

        Args:
            pocket_features: Pocket residue features (P, F)
            pocket_coords: Pocket residue coordinates (P, 3)

        Returns:
            pocket_embedding: Learned representation (P, hidden_dim)
        """
        if pocket_features.shape[0] == 0:
            return torch.zeros(self.hidden_dim, device=pocket_coords.device, dtype=pocket_coords.dtype)

        # Encode scalar features
        x_scalar = self.scalar_proj(pocket_features)  # (P, hidden_dim)

        # Encode vector features
        x_vector = self.vector_proj(pocket_coords)  # (P, hidden_dim)

        # Combine
        x = x_scalar + x_vector  # (P, hidden_dim)

        # Message passing
        for i in range(self.num_layers):
            # Compute pairwise distances for edges (simplified)
            if pocket_coords.shape[0] > 1:
                distances = torch.cdist(pocket_coords, pocket_coords)  # (P, P)
                # Get k-nearest neighbors (k=5)
                k = min(5, pocket_coords.shape[0] - 1)
                _, neighbor_idx = torch.topk(distances, k + 1, dim=1, largest=False)
                neighbor_idx = neighbor_idx[:, 1:]  # Exclude self

                # Simple aggregation: mean of neighbor features
                x_new = x.clone()
                for j, neighbors in enumerate(neighbor_idx):
                    neighbor_features = x[neighbors]  # (k, hidden_dim)
                    aggregated = neighbor_features.mean(dim=0)  # (hidden_dim,)
                    x_new[j] = self.message_layers[i](
                        torch.cat([x[j], aggregated], dim=0)
                    )
                x = x_new

        x = self.output_norm(x)
        pocket_embedding = x.mean(dim=0, keepdim=True)  # Global pooling (1, hidden_dim)
        return pocket_embedding.squeeze(0)  # (hidden_dim,)


class UncertaintyHead(nn.Module):
    """
    Bayesian uncertainty estimation head.

    Outputs mean and variance predictions with confidence intervals.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim

        # Mean prediction
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Variance prediction (log-var for numerical stability)
        self.logvar_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize logvar to be negative (small variance)
        with torch.no_grad():
            self.logvar_head[-1].bias.fill_(-2.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict mean and variance.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            mean: Predicted affinity (batch_size, 1)
            variance: Predicted variance (batch_size, 1)
        """
        mean = self.mean_head(x)  # (batch_size, 1)
        logvar = self.logvar_head(x)  # (batch_size, 1)
        variance = torch.exp(logvar)  # Ensure positive variance
        std = torch.sqrt(variance + 1e-8)
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

        return {
            "mean_affinity": mean,
            "variance": variance,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def get_confidence_interval(
        self, mean: torch.Tensor, variance: torch.Tensor, confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute confidence interval from mean and variance.

        Args:
            mean: Predicted mean (batch_size, 1)
            variance: Predicted variance (batch_size, 1)
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        std = torch.sqrt(variance)

        # Z-score for 95% CI is ~1.96
        z_score = 1.96 if confidence == 0.95 else 1.645
        margin = z_score * std

        lower_bound = mean - margin
        upper_bound = mean + margin

        return lower_bound, upper_bound


class PocketUncertaintyModule(nn.Module):
    """
    Complete pocket-uncertainty module combining all components.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        pocket_radius: float = 8.0,
        use_mc_dropout: bool = False,
        mc_samples: int = 10,
    ):
        super().__init__()
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples

        self.pocket_extractor = PocketExtractor(pocket_radius_angstrom=pocket_radius)
        self.gvp_encoder = GVPEncoder(hidden_dim=hidden_dim)

        # Combine GVP output with other features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.uncertainty_head = UncertaintyHead(hidden_dim)

    def forward(
        self,
        protein_coords: Optional[torch.Tensor],
        protein_features: torch.Tensor,
        ligand_coords: Optional[torch.Tensor],
        other_features: torch.Tensor,
        return_dict: bool = False,
        mc_dropout_samples: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Forward pass with optional pocket extraction.

        Args:
            protein_coords: 3D coordinates (N, 3) or None
            protein_features: Residue features (N, F)
            ligand_coords: Ligand coordinates (M, 3) or None
            other_features: Additional features (batch_size, F_other)

        Returns:
            mean: Predicted affinity
            variance: Uncertainty estimate
            confidence_interval: (lower, upper) confidence bounds
        """
        # If 3D coords available, use GVP on pocket
        if protein_coords is not None and ligand_coords is not None:
            if protein_coords.dim() == 3:
                protein_coords = protein_coords[0]
            if ligand_coords.dim() == 3:
                ligand_coords = ligand_coords[0]
            if protein_features.dim() == 3:
                protein_feature_view = protein_features[0]
            else:
                protein_feature_view = protein_features
            pocket_coords, pocket_feats, _ = self.pocket_extractor(
                protein_coords, ligand_coords, protein_feature_view
            )
            if pocket_feats.numel() == 0:
                gvp_output = protein_feature_view.mean(dim=0, keepdim=True)
            else:
                gvp_output = self.gvp_encoder(pocket_feats, pocket_coords).unsqueeze(0)
            gvp_output = gvp_output.expand(other_features.shape[0], -1)
        else:
            # Fallback: use mean pooling of features
            if protein_features.dim() == 3:
                gvp_output = protein_features.mean(dim=1)
            else:
                gvp_output = protein_features.mean(dim=0, keepdim=True).expand(
                    other_features.shape[0], -1
                )

        # Fuse with other features
        fused = torch.cat([gvp_output, other_features], dim=-1)
        fused = self.fusion(fused)

        # Predict with uncertainty
        if self.use_mc_dropout:
            # Monte Carlo dropout: run multiple forward passes
            means = []
            variances = []
            sample_count = mc_dropout_samples or self.mc_samples
            was_training = self.training
            self.train()
            for _ in range(sample_count):
                sample_outputs = self.uncertainty_head(fused)
                means.append(sample_outputs["mean_affinity"])
                variances.append(sample_outputs["variance"])
            self.train(was_training)
            mean = torch.stack(means).mean(dim=0)
            variance = torch.stack(variances).mean(dim=0)
        else:
            # Standard Bayesian output
            sample_outputs = self.uncertainty_head(fused)
            mean = sample_outputs["mean_affinity"]
            variance = sample_outputs["variance"]

        std = torch.sqrt(variance + 1e-8)
        lower_ci = mean - 1.96 * std
        upper_ci = mean + 1.96 * std
        confidence_interval = torch.cat([lower_ci, upper_ci], dim=-1)
        if return_dict:
            return {
                "mean_affinity": mean,
                "affinity_variance": variance,
                "variance": variance,
                "std": std,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci,
                "confidence_interval": confidence_interval,
                "confidence_interval_95": confidence_interval,
            }

        return mean, variance, (lower_ci, upper_ci)
