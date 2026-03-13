"""
Innovation 2: Joint Pose and Affinity Prediction

Multi-task learning to simultaneously predict:
1. Binding affinity (primary task)
2. Docking pose accuracy (RMSD) (auxiliary task)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformerLayer(nn.Module):
    """
    Transformer layer for heterogeneous graphs.

    Attention over drug atoms and protein residues jointly.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, hidden_dim)
        mask: torch.Tensor,  # (batch, seq_len) - True for valid, False for padding
    ) -> torch.Tensor:
        """
        Apply transformer layer with masking.

        Args:
            x: Node features (batch, seq_len, hidden_dim)
            mask: Attention mask (batch, seq_len)

        Returns:
            output: Transformed features (batch, seq_len, hidden_dim)
        """
        # Create attention mask (inverted for PyTorch convention)
        attn_mask = ~mask if mask.dtype == torch.bool else (mask == 0)

        # Self-attention
        attn_out, _ = self.multihead_attn(
            x, x, x, key_padding_mask=attn_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class UnifiedGraphTransformer(nn.Module):
    """
    Unified transformer encoder for drug-protein heterogeneous graph.

    Drug atoms and protein residues are jointly encoded.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        drug_features: torch.Tensor,  # (batch, n_atoms, hidden_dim)
        drug_mask: torch.Tensor,  # (batch, n_atoms)
        protein_features: torch.Tensor,  # (batch, n_residues, hidden_dim)
        protein_mask: torch.Tensor,  # (batch, n_residues)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Jointly encode drug and protein.

        Args:
            drug_features: Drug atom features
            drug_mask: Valid positions in drug (batch, n_atoms)
            protein_features: Protein residue features
            protein_mask: Valid positions in protein (batch, n_residues)

        Returns:
            drug_output: Encoded drug atoms
            protein_output: Encoded protein residues
            joint_embedding: Pooled joint representation
        """
        batch_size = drug_features.shape[0]

        # Concatenate drug and protein sequences
        joint_input = torch.cat([drug_features, protein_features], dim=1)  # (batch, n_atoms+n_res, dim)
        joint_mask = torch.cat([drug_mask, protein_mask], dim=1)  # (batch, n_atoms+n_res)

        # Apply transformer layers
        for layer in self.layers:
            joint_input = layer(joint_input, joint_mask)

        joint_input = self.output_norm(joint_input)

        # Split back into drug and protein
        drug_len = drug_features.shape[1]
        drug_output = joint_input[:, :drug_len, :]
        protein_output = joint_input[:, drug_len:, :]

        # Global pooling with masking
        drug_pool = self._masked_mean(drug_output, drug_mask)  # (batch, hidden_dim)
        protein_pool = self._masked_mean(protein_output, protein_mask)  # (batch, hidden_dim)
        joint_embedding = torch.cat([drug_pool, protein_pool], dim=-1)  # (batch, 2*hidden_dim)

        return drug_output, protein_output, joint_embedding

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked mean pooling."""
        mask = mask.to(x.dtype).unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)


class AffinityHead(nn.Module):
    """Prediction head for binding affinity."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict affinity score."""
        return self.net(x)


class PoseHead(nn.Module):
    """Prediction head for docking pose (RMSD).
    
    Outputs 7 values: 4 quaternion components (rotation) + 3 translation components.
    RMSD is computed from the translation vector magnitude.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Output 7 values: 4 quaternion + 3 translation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 7),  # 4 quaternion + 3 translation
        )

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """Convert normalized quaternion to rotation matrix."""
        w, x, y, z = quaternion.unbind(dim=-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        row1 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
        row2 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
        row3 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
        return torch.stack([row1, row2, row3], dim=-2)

    def forward(
        self,
        x: torch.Tensor,
        ligand_coords: Optional[torch.Tensor] = None,
        reference_pose: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict pose (rotation + translation).
        
        Args:
            x: Input features (batch, input_dim)
            
        Returns:
            rmsd: Root mean squared deviation (batch,)
            quaternion: Rotation quaternion (batch, 4)
            translation: Translation vector (batch, 3)
        """
        pose_vector = self.net(x)  # (batch, 7)
        
        # Split into quaternion and translation
        quaternion = pose_vector[:, :4]  # (batch, 4)
        translation = pose_vector[:, 4:]  # (batch, 3)
        
        # Normalize quaternion (unit quaternion for rotation)
        quaternion = F.normalize(quaternion, p=2, dim=-1)
        
        rotation = self.quaternion_to_rotation_matrix(quaternion)

        if ligand_coords is not None and reference_pose is not None:
            if ligand_coords.dim() == 2:
                ligand_coords = ligand_coords.unsqueeze(0)
            if reference_pose.dim() == 2:
                reference_pose = reference_pose.unsqueeze(0)
            transformed = torch.matmul(rotation, ligand_coords.transpose(1, 2)).transpose(1, 2)
            transformed = transformed + translation.unsqueeze(1)
            diff = transformed - reference_pose
            rmsd = torch.sqrt(torch.mean(diff.pow(2), dim=(1, 2)) + 1e-8)
        else:
            # Fallback estimate when no reference pose is available.
            rmsd = torch.norm(translation, dim=-1)
        
        # Clamp to reasonable range (0.1 to 10.0 Angstroms)
        rmsd = torch.clamp(rmsd, min=0.1, max=10.0)
        
        return rmsd, quaternion, translation


class MultiTaskPoseModule(nn.Module):
    """
    Multi-task learning module for affinity + pose prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.unified_transformer = UnifiedGraphTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        joint_dim = 2 * hidden_dim

        self.affinity_head = AffinityHead(joint_dim, hidden_dim=256)
        self.pose_head = PoseHead(joint_dim, hidden_dim=256)

    def forward(
        self,
        drug_features: torch.Tensor = None,
        drug_mask: torch.Tensor = None,
        protein_features: torch.Tensor = None,
        protein_mask: torch.Tensor = None,
        drug_embedding: torch.Tensor = None,
        protein_embedding: torch.Tensor = None,
        ligand_coords: Optional[torch.Tensor] = None,
        reference_pose: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Forward pass for multi-task prediction.
        
        Handles both graph-based (drug/protein features) and embedding-based inputs.

        Args:
            drug_features: Drug atom features (batch, n_atoms, hidden_dim) or None
            drug_mask: Valid atom positions or None
            protein_features: Protein residue features (batch, n_residues, hidden_dim) or None
            protein_mask: Valid residue positions or None
            drug_embedding: Pre-computed drug embedding (batch, hidden_dim)
            protein_embedding: Pre-computed protein embedding (batch, hidden_dim or seq_len, hidden_dim)

        Returns:
            affinity_pred: Predicted binding affinity (batch,)
            rmsd: Predicted RMSD (batch,)
            quaternion: Rotation quaternion (batch, 4)
            translation: Translation vector (batch, 3)
            joint_embedding: Combined representation (batch, 2*hidden_dim)
        """
        # Handle embedding-based input (from predict_innovations.py)
        if drug_embedding is not None and protein_embedding is not None:
            # Apply mean pooling to protein if necessary
            if protein_embedding.dim() == 3:
                protein_embedding = protein_embedding.mean(dim=1)  # (batch, seq_len, dim) -> (batch, dim)
            
            # Ensure both are 2D
            if drug_embedding.dim() > 2:
                drug_embedding = drug_embedding.mean(dim=1)
            if protein_embedding.dim() > 2:
                protein_embedding = protein_embedding.mean(dim=1)
            
            # Concatenate embeddings
            joint_embedding = torch.cat([drug_embedding, protein_embedding], dim=-1)  # (batch, 2*hidden_dim)
        
        # Handle graph-based input (from training)
        elif drug_features is not None and protein_features is not None:
            drug_output, protein_output, joint_embedding = self.unified_transformer(
                drug_features, drug_mask, protein_features, protein_mask
            )
        else:
            raise ValueError("Must provide either (drug_features, protein_features) or (drug_embedding, protein_embedding)")

        # Multi-task heads
        affinity_pred = self.affinity_head(joint_embedding).squeeze(-1)  # (batch,)
        rmsd, quaternion, translation = self.pose_head(
            joint_embedding,
            ligand_coords=ligand_coords,
            reference_pose=reference_pose,
        )

        if return_dict:
            return {
                "affinity": affinity_pred,
                "pose_quaternion": quaternion,
                "pose_translation": translation,
                "pose_rmsd": rmsd,
                "joint_embedding": joint_embedding,
            }

        return affinity_pred, rmsd, quaternion, translation, joint_embedding


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining affinity and pose prediction.
    """

    def __init__(self, affinity_weight: float = 1.0, pose_weight: float = 0.4):
        super().__init__()
        self.affinity_weight = affinity_weight
        self.pose_weight = pose_weight
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        affinity_pred: torch.Tensor,
        affinity_true: torch.Tensor,
        pose_pred: torch.Tensor,
        pose_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            affinity_pred: Predicted affinity
            affinity_true: Ground truth affinity
            pose_pred: Predicted RMSD
            pose_true: Ground truth RMSD

        Returns:
            total_loss: Combined loss
            affinity_loss: Affinity MSE
            pose_loss: Pose MSE
        """
        affinity_loss = self.mse_loss(affinity_pred, affinity_true)
        pose_loss = self.mse_loss(pose_pred, pose_true)

        total_loss = self.affinity_weight * affinity_loss + self.pose_weight * pose_loss

        return total_loss, affinity_loss, pose_loss
