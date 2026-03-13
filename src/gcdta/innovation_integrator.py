"""
Integration module for innovations into GCDTA.

Provides wrappers and utilities for seamless innovation integration.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from gcdta.innovations import (
    KnowledgeGraphModule,
    MultiTaskLoss,
    MultiTaskPoseModule,
    PocketUncertaintyModule,
    StructuralNegativesModule,
)


class InnovationIntegrator:
    """
    Integrates innovations into GCDTA training and inference pipelines.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.innovations = config.get("innovations", {})
        self.modules = {}

        # Initialize enabled innovations
        if self.innovations.get("use_pocket_uncertainty", False):
            self._init_pocket_uncertainty()

        if self.innovations.get("use_multitask_pose", False):
            self._init_multitask_pose()

        if self.innovations.get("use_knowledge_graph", False):
            self._init_knowledge_graph()

        if self.innovations.get("use_structural_negatives", False):
            self._init_structural_negatives()

    def _init_pocket_uncertainty(self) -> None:
        """Initialize pocket uncertainty module."""
        params = self.config.get("pocket_uncertainty", {})
        self.modules["pocket_uncertainty"] = PocketUncertaintyModule(
            input_dim=self.config.get("hidden_dim", 128) * 4,
            hidden_dim=params.get("gvp_hidden_dim", 128),
            pocket_radius=params.get("pocket_radius_angstrom", 8.0),
            use_mc_dropout=params.get("use_mc_dropout", False),
            mc_samples=params.get("uncertainty_samples", 10),
        )

    def _init_multitask_pose(self) -> None:
        """Initialize multi-task pose module."""
        params = self.config.get("multitask_pose", {})
        self.modules["multitask_pose"] = MultiTaskPoseModule(
            hidden_dim=self.config.get("hidden_dim", 128),
            num_layers=params.get("transformer_layers", 4),
            num_heads=params.get("transformer_heads", 8),
            dropout=self.config.get("dropout", 0.2),
        )
        self.modules["multitask_loss"] = MultiTaskLoss(
            affinity_weight=self.config.get("affinity_loss_weight", 1.0),
            pose_weight=params.get("pose_weight", 0.4),
        )

    def _init_knowledge_graph(self) -> None:
        """Initialize knowledge graph module."""
        params = self.config.get("knowledge_graph", {})
        self.modules["knowledge_graph"] = KnowledgeGraphModule(
            embedding_dim=params.get("kg_embedding_dim", 128),
            use_graphsage=params.get("use_graphsage", False),
            num_neighbors=params.get("num_neighbors", 5),
            use_semantic_alignment=params.get("use_textual_alignment", True),
        )

    def _init_structural_negatives(self) -> None:
        """Initialize structural negatives module."""
        params = self.config.get("structural_negatives", {})
        self.modules["structural_negatives"] = StructuralNegativesModule(
            temperature=self.config.get("temperature", 0.2),
            use_rmsd_weight=params.get("rmsd_weight", True),
            decoy_ratio=params.get("decoy_ratio", 1.0),
        )

    def get_module(self, name: str) -> nn.Module:
        """Get innovation module by name."""
        return self.modules.get(name)

    def get_all_modules(self) -> Dict[str, nn.Module]:
        """Get all initialized modules."""
        return self.modules

    def is_enabled(self, innovation_name: str) -> bool:
        """Check if innovation is enabled."""
        return self.innovations.get(f"use_{innovation_name}", False)


class EnhancedGCDTA(nn.Module):
    """
    GCDTA model with optional innovation modules.

    Backward compatible with base GCDTA.
    """

    def __init__(self, base_model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.integrator = InnovationIntegrator(config)

        # Register innovation modules
        for name, module in self.integrator.get_all_modules().items():
            if isinstance(module, nn.Module):
                self.add_module(f"innovation_{name}", module)

    def forward(
        self,
        batch: Dict,
        return_innovated: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with optional innovation outputs.

        Args:
            batch: Input batch
            return_innovated: Return innovation outputs

        Returns:
            predictions: Affinity predictions
            cl_loss: Contrastive learning loss
            innovation_outputs: Dict of innovation-specific outputs
        """
        # Base GCDTA forward
        predictions, cl_loss = self.base_model(batch)

        innovation_outputs = {}

        if return_innovated:
            # Pocket Uncertainty
            if self.integrator.is_enabled("pocket_uncertainty"):
                pocket_module = self.integrator.get_module("pocket_uncertainty")
                if pocket_module and batch.get("protein_coords") is not None:
                    mean, variance, ci = pocket_module(
                        protein_coords=batch.get("protein_coords"),
                        protein_features=predictions.unsqueeze(-1),
                        ligand_coords=batch.get("drug_coords"),
                        other_features=batch.get("fused_features", torch.zeros_like(predictions)),
                    )
                    innovation_outputs["uncertainty"] = {
                        "mean": mean,
                        "variance": variance,
                        "confidence_interval": ci,
                    }

            # Multi-task Pose
            if self.integrator.is_enabled("multitask_pose"):
                pose_module = self.integrator.get_module("multitask_pose")
                if pose_module and "drug_features" in batch and "protein_features" in batch:
                    pose_outputs = pose_module(
                        drug_features=batch["drug_features"],
                        drug_mask=batch.get("drug_mask"),
                        protein_features=batch["protein_features"],
                        protein_mask=batch.get("protein_mask"),
                    )
                    # Unpack new 5-tuple: (affinity_pred, rmsd, quaternion, translation, joint_embedding)
                    if isinstance(pose_outputs, tuple) and len(pose_outputs) == 5:
                        affinity_pred, rmsd, quaternion, translation, joint_emb = pose_outputs
                        innovation_outputs["multitask"] = {
                            "affinity": affinity_pred,
                            "pose_rmsd": rmsd,
                            "quaternion": quaternion,
                            "translation": translation,
                            "joint_embedding": joint_emb,
                        }
                    else:
                        # Backward compatibility for old signature
                        affinity_pred, pose_pred = pose_outputs
                        innovation_outputs["multitask"] = {
                            "affinity": affinity_pred,
                            "pose_rmsd": pose_pred,
                        }

            # Knowledge Graph
            if self.integrator.is_enabled("knowledge_graph"):
                kg_module = self.integrator.get_module("knowledge_graph")
                if kg_module and ("drug_id" in batch and "protein_id" in batch or "drug_smiles" in batch and "protein_sequence" in batch):
                    kg_outputs = kg_module(
                        drug_id=batch.get("drug_id"),
                        protein_id=batch.get("protein_id"),
                        drug_smiles=batch.get("drug_smiles"),
                        protein_sequence=batch.get("protein_sequence"),
                        drug_text_embedding=batch.get("drug_text_emb"),
                        protein_text_embedding=batch.get("protein_text_emb"),
                    )
                    # Unpack new 4-tuple: (drug_emb, protein_emb, alignment_loss, kg_similarity)
                    if isinstance(kg_outputs, tuple) and len(kg_outputs) == 4:
                        drug_emb, protein_emb, kg_loss, kg_similarity = kg_outputs
                        innovation_outputs["knowledge_graph"] = {
                            "drug_embedding": drug_emb,
                            "protein_embedding": protein_emb,
                            "alignment_loss": kg_loss,
                            "kg_similarity": kg_similarity,
                        }
                    else:
                        # Backward compatibility for old signature
                        drug_emb, protein_emb, kg_loss = kg_outputs
                        innovation_outputs["knowledge_graph"] = {
                            "drug_embedding": drug_emb,
                            "protein_embedding": protein_emb,
                            "alignment_loss": kg_loss,
                        }

            # Structural Negatives
            if self.integrator.is_enabled("structural_negatives"):
                struct_module = self.integrator.get_module("structural_negatives")
                if struct_module and "negative_embeddings" in batch:
                    struct_outputs = struct_module(
                        anchor_embeddings=batch.get("anchor_emb"),
                        positive_embeddings=batch.get("positive_emb"),
                        negative_embeddings=batch["negative_embeddings"],
                        negative_rmsds=batch.get("negative_rmsds"),
                    )
                    # Unpack new 3-tuple: (loss, pos_similarity, neg_similarity)
                    if isinstance(struct_outputs, tuple) and len(struct_outputs) == 3:
                        struct_loss, pos_sim, neg_sim = struct_outputs
                        innovation_outputs["structural_negatives"] = {
                            "contrastive_loss": struct_loss,
                            "positive_similarity": pos_sim,
                            "negative_similarity": neg_sim,
                        }
                    else:
                        # Backward compatibility for old signature
                        struct_loss = struct_outputs
                        innovation_outputs["structural_negatives"] = {
                            "contrastive_loss": struct_loss,
                        }

        return predictions, cl_loss, innovation_outputs
