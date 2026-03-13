from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from gcdta.config import ConfigLoader
from gcdta.innovation_integrator import EnhancedGCDTA
from gcdta.runtime import load_checkpoint
from gcdta.train_utils import to_device


PATH_SPECS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("base", None, "Baseline"),
    ("path1", "path1_pocket_uncertainty.yaml", "Pocket Uncertainty"),
    ("path2", "path2_multitask_pose.yaml", "Multitask Pose"),
    ("path3", "path3_knowledge_graph.yaml", "Knowledge Graph"),
    ("path4", "path4_structural_negatives.yaml", "Structural Negatives"),
)


@dataclass
class PathSummary:
    path: str
    label: str
    affinity: float
    uncertainty_variance: Optional[float]
    uncertainty_std: Optional[float]
    uncertainty_ci_lower: Optional[float]
    uncertainty_ci_upper: Optional[float]
    pose_rmsd: Optional[float]
    kg_adjusted_affinity: Optional[float]
    kg_similarity: Optional[float]
    kg_top_neighbors: Optional[List[Tuple[str, float]]]
    structural_contrastive_loss: Optional[float]
    contrastive_loss: float
    processing_time_seconds: float
    extra: str


def resolve_config_path(config_name: Optional[str]) -> Optional[Path]:
    if config_name is None:
        return None
    candidate = Path(config_name)
    if candidate.exists():
        return candidate
    config_root = Path(__file__).resolve().parents[2] / "configs"
    if candidate.suffix:
        return config_root / candidate.name
    yaml_path = config_root / f"{candidate.name}.yaml"
    if yaml_path.exists():
        return yaml_path
    yml_path = config_root / f"{candidate.name}.yml"
    if yml_path.exists():
        return yml_path
    return config_root / candidate.name


def load_model_for_config(
    model_path: Path,
    config_name: Optional[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    model, loaded_config = load_checkpoint(model_path, device=device)
    config_path = resolve_config_path(config_name)
    if config_path is None:
        return model, loaded_config

    config_loader = ConfigLoader()
    config = config_loader.load(str(config_path))
    if any(config.get("innovations", {}).values()):
        model = EnhancedGCDTA(model, config).to(device)
    return model, config


def _unwrap_base_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.base_model if isinstance(model, EnhancedGCDTA) else model


def _extract_batch_string(batch: Mapping[str, Any], key: str) -> Optional[str]:
    value = batch.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return None


def _mean_optional(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return float(value)
    detached = value.detach().float().reshape(-1)
    if detached.numel() == 0:
        return None
    return float(detached.mean().cpu().item())


def _prepare_backbone_outputs(model: torch.nn.Module, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    base_model = _unwrap_base_model(model)
    drug_tokens, drug_mask, drug_graph = base_model.drug_encoder(batch["drug_graph"])
    target_tokens, target_graph = base_model.target_encoder(
        token_ids=batch["target_tokens"],
        physchem=batch["target_physchem"],
        target_mask=batch["target_mask"],
    )
    target_mask = batch["target_mask"] > 0
    fused = base_model.fusion(
        drug_tokens=drug_tokens,
        drug_mask=drug_mask,
        target_tokens=target_tokens,
        target_mask=target_mask,
        drug_graph=drug_graph,
        target_graph=target_graph,
    )
    cl_loss, drug_ctx, target_ctx = base_model.hgcn_contrastive(
        drug_emb=drug_graph,
        target_emb=target_graph,
        drug_ids=batch["drug_node_id"],
        target_ids=batch["target_node_id"],
    )
    pred_input = torch.cat([fused, drug_ctx, target_ctx], dim=-1)
    affinity = base_model.regressor(pred_input).squeeze(-1)
    return {
        "affinity": affinity,
        "contrastive_loss": cl_loss,
        "drug_tokens": drug_tokens,
        "drug_mask": drug_mask,
        "drug_graph": drug_graph,
        "target_tokens": target_tokens,
        "target_mask": target_mask,
        "target_graph": target_graph,
        "fused": fused,
        "drug_ctx": drug_ctx,
        "target_ctx": target_ctx,
    }


def run_innovation_forward(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    backbone = _prepare_backbone_outputs(model, batch)
    results: Dict[str, Any] = {
        "affinity": backbone["affinity"],
        "contrastive_loss": backbone["contrastive_loss"],
    }

    if not isinstance(model, EnhancedGCDTA):
        return results

    integrator = model.integrator

    if integrator.is_enabled("pocket_uncertainty"):
        pocket_module = integrator.get_module("pocket_uncertainty")
        if pocket_module is not None:
            base_affinity = results["affinity"]
            pocket_outputs = pocket_module(
                protein_coords=batch.get("protein_coords"),
                protein_features=backbone["target_tokens"],
                ligand_coords=batch.get("ligand_coords", batch.get("drug_coords")),
                other_features=backbone["fused"],
                return_dict=True,
            )
            raw_mean = pocket_outputs["mean_affinity"].squeeze(-1)
            variance = pocket_outputs["affinity_variance"]
            std = pocket_outputs["std"].squeeze(-1)
            calibrated_mean = 0.8 * base_affinity + 0.2 * raw_mean
            ci_lower = calibrated_mean - 1.96 * std
            ci_upper = calibrated_mean + 1.96 * std
            ci = torch.stack([ci_lower, ci_upper], dim=-1)
            results["affinity"] = calibrated_mean
            results["uncertainty_mean"] = calibrated_mean
            results["uncertainty_variance"] = variance.squeeze(-1)
            results["uncertainty_std"] = std
            results["uncertainty_ci_lower"] = ci_lower
            results["uncertainty_ci_upper"] = ci_upper
            results["uncertainty_ci"] = ci

    if integrator.is_enabled("multitask_pose"):
        pose_module = integrator.get_module("multitask_pose")
        if pose_module is not None:
            pose_outputs = pose_module(
                drug_features=backbone["drug_tokens"],
                drug_mask=backbone["drug_mask"],
                protein_features=backbone["target_tokens"],
                protein_mask=backbone["target_mask"],
                ligand_coords=batch.get("ligand_coords", batch.get("drug_coords")),
                reference_pose=batch.get("reference_pose"),
                return_dict=True,
            )
            pose_affinity = pose_outputs["affinity"]
            pose_rmsd = pose_outputs["pose_rmsd"]
            quaternion = pose_outputs["pose_quaternion"]
            translation = pose_outputs["pose_translation"]
            joint_embedding = pose_outputs["joint_embedding"]
            results["affinity"] = 0.8 * results["affinity"] + 0.2 * pose_affinity
            results["multitask_affinity"] = pose_affinity
            results["pose_rmsd"] = pose_rmsd
            results["pose_quaternion"] = quaternion
            results["pose_translation"] = translation
            results["joint_embedding"] = joint_embedding

    if integrator.is_enabled("knowledge_graph"):
        kg_module = integrator.get_module("knowledge_graph")
        if kg_module is not None:
            smiles = _extract_batch_string(batch, "smiles")
            fasta = _extract_batch_string(batch, "fasta")
            kg_outputs = kg_module(
                drug_smiles=smiles,
                protein_sequence=fasta,
                drug_id=batch.get("drug_id", batch.get("drug_node_id")),
                protein_id=batch.get("protein_id", batch.get("target_node_id")),
                drug_text_embedding=batch.get("drug_text_emb"),
                protein_text_embedding=batch.get("protein_text_emb"),
                return_dict=True,
            )
            drug_embedding = kg_outputs["drug_embedding"]
            protein_embedding = kg_outputs["protein_embedding"]
            alignment_loss = kg_outputs["alignment_loss"]
            kg_similarity = kg_outputs["kg_similarity"]
            dt_embedding = torch.cat([backbone["drug_ctx"], backbone["target_ctx"]], dim=-1)
            kg_embedding = torch.cat([drug_embedding, protein_embedding], dim=-1).to(dt_embedding.device)
            fused_with_kg = torch.cat([backbone["fused"], kg_embedding], dim=-1)
            kg_gate = 0.5 * F.cosine_similarity(dt_embedding, kg_embedding, dim=-1) + 0.5 * torch.tanh(
                fused_with_kg.mean(dim=-1)
            )
            adjusted_affinity = (
                results["affinity"]
                + 0.05 * (kg_similarity.to(results["affinity"].device) - 0.5)
                + 0.05 * kg_gate
            )
            results["affinity"] = adjusted_affinity
            results["kg_adjusted_affinity"] = adjusted_affinity
            results["kg_similarity"] = kg_similarity
            results["knowledge_graph_alignment_loss"] = alignment_loss
            results["kg_drug_embedding"] = drug_embedding
            results["kg_protein_embedding"] = protein_embedding
            results["kg_fused_embedding"] = fused_with_kg
            results["kg_top_neighbors"] = kg_outputs.get("top_neighbors", [])

    if integrator.is_enabled("structural_negatives"):
        struct_module = integrator.get_module("structural_negatives")
        if struct_module is not None:
            negative_embeddings = batch.get("negative_embeddings")
            if negative_embeddings is None and backbone["target_graph"].shape[0] > 1:
                negative_embeddings = backbone["target_graph"].roll(shifts=1, dims=0)
            struct_outputs = struct_module(
                anchor_embeddings=backbone["drug_graph"],
                positive_embeddings=backbone["target_graph"],
                negative_embeddings=negative_embeddings,
                negative_rmsds=batch.get("negative_rmsds"),
                affinity=results["affinity"],
                return_dict=True,
            )
            struct_loss = struct_outputs["contrastive_loss"]
            pos_similarity = struct_outputs["positive_similarity"]
            neg_similarity = struct_outputs["negative_similarity"]
            similarity_margin = (pos_similarity - neg_similarity).to(results["affinity"].device)
            results["affinity"] = results["affinity"] + 0.1 * similarity_margin
            results["structural_contrastive_loss"] = struct_loss
            results["structural_positive_similarity"] = pos_similarity
            results["structural_negative_similarity"] = neg_similarity

    return results


def summarize_path_output(
    path: str,
    label: str,
    outputs: Mapping[str, Any],
    elapsed: float,
) -> PathSummary:
    kg_adjusted_affinity = _mean_optional(outputs.get("kg_adjusted_affinity"))
    uncertainty_variance = _mean_optional(outputs.get("uncertainty_variance"))
    uncertainty_std = _mean_optional(outputs.get("uncertainty_std"))
    uncertainty_ci_lower = _mean_optional(outputs.get("uncertainty_ci_lower"))
    uncertainty_ci_upper = _mean_optional(outputs.get("uncertainty_ci_upper"))
    pose_rmsd = _mean_optional(outputs.get("pose_rmsd"))
    kg_similarity = _mean_optional(outputs.get("kg_similarity"))
    structural_loss = _mean_optional(outputs.get("structural_contrastive_loss"))
    kg_top_neighbors = outputs.get("kg_top_neighbors")

    if uncertainty_ci_lower is not None and uncertainty_ci_upper is not None:
        extra = f"CI=[{uncertainty_ci_lower:.2f},{uncertainty_ci_upper:.2f}]"
    elif pose_rmsd is not None:
        extra = f"RMSD={pose_rmsd:.4f}"
    elif kg_similarity is not None:
        extra = f"sim={kg_similarity:.2f}"
    elif structural_loss is not None:
        extra = f"cl_loss={structural_loss:.5f}"
    else:
        extra = "-"

    return PathSummary(
        path=path,
        label=label,
        affinity=_mean_optional(outputs.get("affinity")) or 0.0,
        uncertainty_variance=uncertainty_variance,
        uncertainty_std=uncertainty_std,
        uncertainty_ci_lower=uncertainty_ci_lower,
        uncertainty_ci_upper=uncertainty_ci_upper,
        pose_rmsd=pose_rmsd,
        kg_adjusted_affinity=kg_adjusted_affinity,
        kg_similarity=kg_similarity,
        kg_top_neighbors=kg_top_neighbors,
        structural_contrastive_loss=structural_loss,
        contrastive_loss=_mean_optional(outputs.get("contrastive_loss")) or 0.0,
        processing_time_seconds=elapsed,
        extra=extra,
    )


@torch.no_grad()
def run_innovation_paths(
    model_path: Path,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, PathSummary]:
    results: Dict[str, PathSummary] = {}
    for path, config_name, label in PATH_SPECS:
        model, config = load_model_for_config(model_path, config_name, device)
        model.eval()
        start = perf_counter()
        outputs = run_innovation_forward(model, to_device(dict(batch), device), config)
        elapsed = perf_counter() - start
        results[path] = summarize_path_output(path, label, outputs, elapsed)
    return results


@torch.no_grad()
def evaluate_innovation_paths(
    model_path: Path,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    from gcdta.metrics import regression_metrics

    aggregate: Dict[str, Dict[str, Any]] = {}
    for path, config_name, label in PATH_SPECS:
        model, config = load_model_for_config(model_path, config_name, device)
        model.eval()
        all_true: List[torch.Tensor] = []
        all_pred: List[torch.Tensor] = []
        uncertainty_values: List[torch.Tensor] = []
        pose_values: List[torch.Tensor] = []
        kg_values: List[torch.Tensor] = []
        structural_values: List[torch.Tensor] = []
        total_time = 0.0

        for batch in loader:
            start = perf_counter()
            device_batch = to_device(batch, device)
            outputs = run_innovation_forward(model, device_batch, config)
            total_time += perf_counter() - start
            all_true.append(device_batch["affinity"].detach().cpu())
            all_pred.append(outputs["affinity"].detach().cpu())

            if outputs.get("uncertainty_variance") is not None:
                uncertainty_values.append(outputs["uncertainty_variance"].detach().cpu().reshape(-1))
            if outputs.get("pose_rmsd") is not None:
                pose_values.append(outputs["pose_rmsd"].detach().cpu().reshape(-1))
            if outputs.get("kg_similarity") is not None:
                kg_values.append(outputs["kg_similarity"].detach().cpu().reshape(-1))
            if outputs.get("structural_contrastive_loss") is not None:
                structural_values.append(outputs["structural_contrastive_loss"].detach().cpu().reshape(-1))

        y_true = torch.cat(all_true).numpy()
        y_pred = torch.cat(all_pred).numpy()
        metrics = regression_metrics(y_true, y_pred)
        outputs_for_summary: Dict[str, Any] = {
            "affinity": torch.tensor(y_pred.mean()),
            "contrastive_loss": torch.tensor(0.0),
        }
        if uncertainty_values:
            outputs_for_summary["uncertainty_variance"] = torch.cat(uncertainty_values)
        if pose_values:
            outputs_for_summary["pose_rmsd"] = torch.cat(pose_values)
        if kg_values:
            outputs_for_summary["kg_similarity"] = torch.cat(kg_values)
            outputs_for_summary["kg_adjusted_affinity"] = torch.tensor(y_pred.mean())
        if structural_values:
            outputs_for_summary["structural_contrastive_loss"] = torch.cat(structural_values)

        aggregate[path] = {
            "label": label,
            "metrics": metrics,
            "summary": summarize_path_output(path, label, outputs_for_summary, total_time),
        }
    return aggregate


def format_comparison_table(results: Mapping[str, PathSummary]) -> str:
    header = f"{'Path':<10} {'Affinity':<12} {'Uncertainty':<14} {'Extra':<24} {'Time':<10}"
    rows = [header, "-" * len(header)]
    for _, summary in results.items():
        uncertainty = (
            f"var={summary.uncertainty_variance:.4f}"
            if summary.uncertainty_variance is not None
            else "-"
        )
        rows.append(
            f"{summary.path:<10} {summary.affinity:<12.4f} {uncertainty:<14} {summary.extra:<24} {summary.processing_time_seconds:<10.4f}"
        )
    return "\n".join(rows)
