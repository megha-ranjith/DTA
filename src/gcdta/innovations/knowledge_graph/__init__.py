"""
Innovation 3: Knowledge Graph Zero-Shot Learning

Enable zero-shot prediction for unseen drugs/proteins using
external knowledge graphs and semantic alignment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    RDLogger = None


class KGLoaderBase(nn.Module):
    """Base class for knowledge graph loaders."""

    def __init__(self, kg_embedding_dim: int = 128):
        super().__init__()
        self.kg_embedding_dim = kg_embedding_dim

    def load(self) -> Tuple[Dict, Dict]:
        """Load KG entities and relations. Return entity_embeddings, relation_embeddings."""
        raise NotImplementedError


class TransEEmbedding(nn.Module):
    """
    Simplified TransE embeddings for KG.

    TransE models: h + r ≈ t (head + relation ≈ tail)
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Embeddings for common biomedical entities
        self.drug_embedding = nn.Embedding(50000, embedding_dim)  # Large vocab
        self.protein_embedding = nn.Embedding(100000, embedding_dim)  # Proteins
        self.disease_embedding = nn.Embedding(20000, embedding_dim)
        self.pathway_embedding = nn.Embedding(10000, embedding_dim)

        # Relations
        self.relation_embedding = nn.Embedding(10, embedding_dim)  # 10 relation types

        # Initialize
        nn.init.uniform_(self.drug_embedding.weight, -1, 1)
        nn.init.uniform_(self.protein_embedding.weight, -1, 1)
        nn.init.uniform_(self.disease_embedding.weight, -1, 1)
        nn.init.uniform_(self.pathway_embedding.weight, -1, 1)
        nn.init.uniform_(self.relation_embedding.weight, -1, 1)

    def forward(
        self,
        entity_type: str,
        entity_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get embedding for entity.

        Args:
            entity_type: "drug", "protein", "disease", or "pathway"
            entity_id: Entity indices

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        if entity_type == "drug":
            return F.normalize(self.drug_embedding(entity_id), p=2, dim=-1)
        elif entity_type == "protein":
            return F.normalize(self.protein_embedding(entity_id), p=2, dim=-1)
        elif entity_type == "disease":
            return F.normalize(self.disease_embedding(entity_id), p=2, dim=-1)
        elif entity_type == "pathway":
            return F.normalize(self.pathway_embedding(entity_id), p=2, dim=-1)
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")


class GraphSAGEAggregator(nn.Module):
    """
    Simplified GraphSAGE for neighbor aggregation in KG.
    """

    def __init__(self, embedding_dim: int = 128, num_neighbors: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_neighbors = num_neighbors

        self.aggregator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        node_embedding: torch.Tensor,  # (batch, embedding_dim)
        neighbor_embeddings: torch.Tensor,  # (batch, num_neighbors, embedding_dim)
    ) -> torch.Tensor:
        """
        Aggregate node and neighbor embeddings.

        Args:
            node_embedding: Center node embedding
            neighbor_embeddings: Neighbor embeddings

        Returns:
            aggregated: Updated node embedding
        """
        # Mean aggregation of neighbors
        neighbor_mean = neighbor_embeddings.mean(dim=1)  # (batch, embedding_dim)

        # Concatenate and aggregate
        combined = torch.cat([node_embedding, neighbor_mean], dim=-1)  # (batch, 2*dim)
        aggregated = self.aggregator(combined)

        return aggregated


class SemanticAlignmentModel(nn.Module):
    """
    Align KG embeddings with textual/LLM embeddings for semantic understanding.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Simple projection from text embeddings to KG space
        self.text_projector = nn.Sequential(
            nn.Linear(768, embedding_dim),  # PubMedBERT uses 768-dim
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        kg_embedding: torch.Tensor,  # (batch, embedding_dim)
        text_embedding: torch.Tensor,  # (batch, 768) from PubMedBERT
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project text to KG space and compute alignment.

        Args:
            kg_embedding: KG embedding
            text_embedding: Text/LLM embedding

        Returns:
            projected_text: Projected text embedding
            similarity: Cosine similarity
        """
        projected_text = self.text_projector(text_embedding)
        projected_text = F.normalize(projected_text, p=2, dim=-1)
        kg_embedding = F.normalize(kg_embedding, p=2, dim=-1)

        similarity = (projected_text * kg_embedding).sum(dim=-1)  # Cosine sim

        return projected_text, similarity


class AlignmentLoss(nn.Module):
    """
    Contrastive loss for semantic alignment.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_embeddings: torch.Tensor,  # (batch, embedding_dim)
        kg_embeddings: torch.Tensor,  # (batch, embedding_dim)
    ) -> torch.Tensor:
        """
        Compute contrastive alignment loss.

        Positive pairs: aligned text-KG embeddings
        Negative pairs: misaligned pairs
        """
        # Normalize
        text_emb = F.normalize(text_embeddings, p=2, dim=-1)
        kg_emb = F.normalize(kg_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        logits = (text_emb @ kg_emb.T) / self.temperature  # (batch, batch)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Symmetric contrastive loss
        loss_forward = F.cross_entropy(logits, labels)
        loss_backward = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_forward + loss_backward)


class KnowledgeGraphModule(nn.Module):
    """
    Complete KG module for zero-shot learning.
    
    Only activates for unseen drugs/proteins.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        use_graphsage: bool = False,
        num_neighbors: int = 5,
        use_semantic_alignment: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_graphsage = use_graphsage

        # KG embeddings
        self.transE = TransEEmbedding(embedding_dim=embedding_dim)

        # GraphSAGE aggregator
        if use_graphsage:
            self.aggregator = GraphSAGEAggregator(embedding_dim, num_neighbors)

        # Semantic alignment (text to KG)
        if use_semantic_alignment:
            self.semantic_model = SemanticAlignmentModel(embedding_dim)
            self.alignment_loss = AlignmentLoss()
            
        # Store training set for zero-shot detection
        self.training_drugs = set()
        self.training_proteins = set()
        self.kg_affinity_matrix = {}  # Store known affinities for interpolation
        self.project_root = Path(__file__).resolve().parents[4]
        self.processed_root = self.project_root / "data" / "processed"
        self.fingerprint_db_path = self.processed_root / "drug_fingerprints.pkl"
        self.fingerprint_db: Dict[str, object] = {}
        self.drug_smiles_lookup: Dict[str, str] = {}
        self.kg_embeddings: Dict[str, torch.Tensor] = {}
        self._load_or_build_fingerprint_db()

    def register_training_data(self, drugs: set, proteins: set, affinity_matrix: Dict = None):
        """
        Register training data for zero-shot detection.
        
        Args:
            drugs: Set of training drug SMILES
            proteins: Set of training protein sequences
            affinity_matrix: Dictionary of known drug-protein affinities
        """
        self.training_drugs = drugs
        self.training_proteins = proteins
        if affinity_matrix:
            self.kg_affinity_matrix = affinity_matrix

    @staticmethod
    def _stable_index(key: str, modulo: int) -> int:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % modulo

    def _load_or_build_fingerprint_db(self) -> None:
        if self.fingerprint_db_path.exists():
            try:
                payload = pickle.loads(self.fingerprint_db_path.read_bytes())
                self.fingerprint_db = payload.get("fingerprints", {})
                self.drug_smiles_lookup = payload.get("smiles", {})
            except Exception:
                self.fingerprint_db = {}
                self.drug_smiles_lookup = {}

        if self.fingerprint_db:
            self._initialize_kg_embeddings()
            return

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except Exception:
            return

        fingerprints: Dict[str, object] = {}
        smiles_lookup: Dict[str, str] = {}
        for csv_path in sorted(self.processed_root.glob("*.csv")):
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        drug_id = str(row.get("drug_id") or row.get("smiles") or "").strip()
                        smiles = str(row.get("smiles") or "").strip()
                        if not drug_id or not smiles or drug_id in fingerprints:
                            continue
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            continue
                        fingerprints[drug_id] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        smiles_lookup[drug_id] = smiles
            except Exception:
                continue

        self.fingerprint_db = fingerprints
        self.drug_smiles_lookup = smiles_lookup
        if fingerprints:
            self.processed_root.mkdir(parents=True, exist_ok=True)
            payload = {"fingerprints": fingerprints, "smiles": smiles_lookup}
            with self.fingerprint_db_path.open("wb") as handle:
                pickle.dump(payload, handle)
        self._initialize_kg_embeddings()

    def _initialize_kg_embeddings(self) -> None:
        embeddings: Dict[str, torch.Tensor] = {}
        for drug_id in self.fingerprint_db:
            idx = self._stable_index(str(drug_id), self.transE.drug_embedding.num_embeddings)
            emb = self.transE("drug", torch.tensor([idx], dtype=torch.long)).squeeze(0).detach().cpu()
            embeddings[str(drug_id)] = emb
        self.kg_embeddings = embeddings

    def get_drug_fingerprint(self, smiles: str) -> torch.Tensor:
        """
        Compute Morgan fingerprint for a drug SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            fingerprint: Binary fingerprint vector (128,)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(2048)
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return torch.tensor(list(fp), dtype=torch.float32)
        except Exception:
            # Deterministic fallback when RDKit is unavailable.
            return torch.zeros(2048)

    def _fingerprint_from_smiles(self, smiles: str):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except Exception:
            return None

    def find_similar_drugs_in_kg(self, drug_smiles: str, k: int = 5) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
        """
        Find k most similar drugs in KG using fingerprints.
        
        Args:
            drug_smiles: Query drug SMILES
            k: Number of neighbors to retrieve
            
        Returns:
            similar_drugs: List of tuples (drug_id_or_smiles, similarity)
            similarities: Similarity scores (k,)
        """
        query_fp = self._fingerprint_from_smiles(drug_smiles)
        if query_fp is None or not self.fingerprint_db:
            return [], torch.zeros(k, dtype=torch.float32)

        try:
            from rdkit import DataStructs
        except Exception:
            return [], torch.zeros(k, dtype=torch.float32)

        similarities: List[Tuple[str, float]] = []
        for drug_id, train_fp in self.fingerprint_db.items():
            sim = float(DataStructs.TanimotoSimilarity(query_fp, train_fp))
            label = self.drug_smiles_lookup.get(str(drug_id), str(drug_id))
            similarities.append((label, sim))

        similarities.sort(key=lambda item: item[1], reverse=True)
        top_neighbors = similarities[:k]
        sim_values = torch.tensor([score for _, score in top_neighbors], dtype=torch.float32)
        if sim_values.numel() < k:
            sim_values = F.pad(sim_values, (0, k - sim_values.numel()))
        return top_neighbors, sim_values

    def get_kg_embedding(self, drug_smiles: str, k: int = 5) -> Tuple[torch.Tensor, float, List[Tuple[str, float]]]:
        neighbors, sim_values = self.find_similar_drugs_in_kg(drug_smiles, k=k)
        if not neighbors:
            fallback_id = self._stable_index(drug_smiles, self.transE.drug_embedding.num_embeddings)
            fallback_emb = self.transE("drug", torch.tensor([fallback_id], dtype=torch.long)).squeeze(0).detach()
            return fallback_emb, 0.0, []

        weights = torch.softmax(sim_values[: len(neighbors)], dim=0)
        emb = torch.zeros(self.embedding_dim, dtype=torch.float32)
        for weight, (drug_label, _) in zip(weights, neighbors):
            embedding = self.kg_embeddings.get(drug_label)
            if embedding is None:
                stable_idx = self._stable_index(drug_label, self.transE.drug_embedding.num_embeddings)
                embedding = self.transE("drug", torch.tensor([stable_idx], dtype=torch.long)).squeeze(0).detach().cpu()
            emb = emb + weight * embedding.float()

        return emb, neighbors[0][1], neighbors

    def forward(
        self,
        drug_smiles: str = None,
        protein_sequence: str = None,
        drug_id: torch.Tensor = None,
        protein_id: torch.Tensor = None,
        drug_text_embedding: Optional[torch.Tensor] = None,
        protein_text_embedding: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
        Dict[str, Union[torch.Tensor, float, List[Tuple[str, float]]]],
    ]:
        """
        Forward pass for KG zero-shot learning.

        Args:
            drug_smiles: Drug SMILES string (for zero-shot detection)
            protein_sequence: Protein sequence (for zero-shot detection)
            drug_id: Drug entity IDs (batch,) [deprecated, for backward compat]
            protein_id: Protein entity IDs (batch,) [deprecated, for backward compat]
            drug_text_embedding: Text embeddings for drugs (batch, 768) or None
            protein_text_embedding: Text embeddings for proteins (batch, 768) or None

        Returns:
            drug_embedding: Drug KG embedding (batch, embedding_dim)
            protein_embedding: Protein KG embedding (batch, embedding_dim)
            alignment_loss: Optional loss for semantic alignment
            kg_similarity: KG similarity score (batch,)
        """
        top_neighbors: List[Tuple[str, float]] = []
        kg_similarity = torch.tensor([0.0], dtype=torch.float32)

        if drug_smiles is not None:
            drug_kg_from_neighbors, top_similarity, top_neighbors = self.get_kg_embedding(drug_smiles)
            kg_similarity = torch.tensor([top_similarity], dtype=torch.float32)
        else:
            drug_kg_from_neighbors = None

        if drug_id is None:
            raw_drug_key = drug_smiles or "unknown_drug"
            drug_id = torch.tensor([self._stable_index(raw_drug_key, self.transE.drug_embedding.num_embeddings)], dtype=torch.long)
        if protein_id is None:
            raw_protein_key = protein_sequence or "unknown_protein"
            protein_id = torch.tensor([self._stable_index(raw_protein_key, self.transE.protein_embedding.num_embeddings)], dtype=torch.long)
        
        # Ensure batch size matches
        if isinstance(drug_id, torch.Tensor) and drug_id.dim() == 0:
            drug_id = drug_id.unsqueeze(0)
        if isinstance(protein_id, torch.Tensor) and protein_id.dim() == 0:
            protein_id = protein_id.unsqueeze(0)
        
        batch_size = max(
            drug_id.shape[0] if isinstance(drug_id, torch.Tensor) else 1,
            protein_id.shape[0] if isinstance(protein_id, torch.Tensor) else 1
        )
        
        # Get KG embeddings (TransE)
        drug_kg_emb = self.transE("drug", drug_id)  # (batch, dim)
        protein_kg_emb = self.transE("protein", protein_id)  # (batch, dim)
        if drug_kg_from_neighbors is not None:
            neighbor_emb = drug_kg_from_neighbors.to(drug_kg_emb.device).unsqueeze(0).expand(batch_size, -1)
            drug_kg_emb = 0.5 * drug_kg_emb + 0.5 * neighbor_emb

        # Apply GraphSAGE if enabled
        if self.use_graphsage and hasattr(self, "aggregator"):
            dummy_neighbors = drug_kg_emb.unsqueeze(1).expand(batch_size, 5, self.embedding_dim)
            drug_kg_emb = self.aggregator(drug_kg_emb, dummy_neighbors)

        # Compute alignment loss if text embeddings provided
        alignment_loss = None
        if drug_text_embedding is not None and hasattr(self, "semantic_model"):
            if protein_text_embedding is not None:
                aligned_drug_text = self.semantic_model.text_projector(drug_text_embedding)
                aligned_protein_text = self.semantic_model.text_projector(protein_text_embedding)
                alignment_loss = self.alignment_loss(
                    torch.cat([aligned_drug_text, aligned_protein_text], dim=0),
                    torch.cat([drug_kg_emb, protein_kg_emb], dim=0),
                )
        
        # Ensure kg_similarity is on same device and correct batch size
        if kg_similarity.shape[0] < batch_size:
            kg_similarity = kg_similarity.repeat(batch_size)
        elif kg_similarity.shape[0] > batch_size:
            kg_similarity = kg_similarity[:batch_size]

        if return_dict:
            return {
                "drug_embedding": drug_kg_emb,
                "protein_embedding": protein_kg_emb,
                "alignment_loss": alignment_loss,
                "kg_similarity": kg_similarity,
                "top_neighbors": top_neighbors,
            }

        return drug_kg_emb, protein_kg_emb, alignment_loss, kg_similarity

    def zero_shot_prediction(
        self,
        known_drug_embeddings: torch.Tensor,  # (n_known, dim)
        known_protein_embeddings: torch.Tensor,  # (m_known, dim)
        unseen_drug_embedding: torch.Tensor,  # (dim,)
        unseen_protein_embedding: torch.Tensor,  # (dim,)
        known_affinities: torch.Tensor,  # (n_known, m_known)
    ) -> torch.Tensor:
        """
        Predict affinity for unseen drug-protein pair using KG neighbors.

        Strategy: Find k-nearest known drugs/proteins and interpolate affinity.

        Args:
            known_drug_embeddings: Known drug KG embeddings
            known_protein_embeddings: Known protein KG embeddings
            unseen_drug_embedding: New drug KG embedding
            unseen_protein_embedding: New protein KG embedding
            known_affinities: Matrix of known affinities

        Returns:
            predicted_affinity: Zero-shot prediction
        """
        k = min(3, known_drug_embeddings.shape[0])  # K-nearest

        # Find similar drugs
        drug_sims = F.cosine_similarity(
            unseen_drug_embedding.unsqueeze(0), known_drug_embeddings, dim=-1
        )
        top_drug_indices = torch.topk(drug_sims, k).indices

        # Find similar proteins
        protein_sims = F.cosine_similarity(
            unseen_protein_embedding.unsqueeze(0), known_protein_embeddings, dim=-1
        )
        top_protein_indices = torch.topk(protein_sims, k).indices

        # Average affinity from similar pairs
        affinity_estimates = []
        for drug_idx in top_drug_indices:
            for protein_idx in top_protein_indices:
                aff = known_affinities[drug_idx, protein_idx]
                if not torch.isnan(aff):
                    affinity_estimates.append(aff)

        if affinity_estimates:
            predicted_affinity = torch.stack(affinity_estimates).mean()
        else:
            predicted_affinity = torch.tensor(6.8)  # Default to Davis mean

        return predicted_affinity
