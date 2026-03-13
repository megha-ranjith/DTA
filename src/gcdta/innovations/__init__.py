"""
GCDTA Innovations Package

Contains 4 innovations for improved drug-target affinity prediction:

1. PocketUncertaintyModule - 3D pocket extraction with uncertainty estimation
2. MultiTaskPoseModule - Joint affinity and pose prediction
3. KnowledgeGraphModule - Zero-shot learning using external KGs
4. StructuralNegativesModule - Hard negative training with decoys
"""

from .pocket_uncertainty import (
    GVPEncoder,
    PocketExtractor,
    PocketUncertaintyModule,
    UncertaintyHead,
)
from .multitask_pose import (
    AffinityHead,
    MultiTaskLoss,
    MultiTaskPoseModule,
    PoseHead,
    UnifiedGraphTransformer,
)
from .knowledge_graph import (
    AlignmentLoss,
    GraphSAGEAggregator,
    KnowledgeGraphModule,
    SemanticAlignmentModel,
    TransEEmbedding,
)
from .structural_negatives import (
    DecoySampler,
    StructuralContrastiveLoss,
    StructuralNegativesModule,
    StructuralPerturbation,
)

__all__ = [
    # Pocket Uncertainty
    "PocketExtractor",
    "GVPEncoder",
    "UncertaintyHead",
    "PocketUncertaintyModule",
    # Multi-task Pose
    "UnifiedGraphTransformer",
    "AffinityHead",
    "PoseHead",
    "MultiTaskPoseModule",
    "MultiTaskLoss",
    # Knowledge Graph
    "TransEEmbedding",
    "GraphSAGEAggregator",
    "SemanticAlignmentModel",
    "AlignmentLoss",
    "KnowledgeGraphModule",
    # Structural Negatives
    "DecoySampler",
    "StructuralPerturbation",
    "StructuralContrastiveLoss",
    "StructuralNegativesModule",
]
