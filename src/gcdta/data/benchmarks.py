from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


GRAPH_DTA_BASE = "https://raw.githubusercontent.com/thinng/GraphDTA/master/data"
DATADTA_BASE = "https://raw.githubusercontent.com/YanZhu06/DataDTA/master/data"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    raw_files: Dict[str, str]
    description: str


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "davis": BenchmarkSpec(
        name="davis",
        raw_files={
            "ligands_can.txt": f"{GRAPH_DTA_BASE}/davis/ligands_can.txt",
            "proteins.txt": f"{GRAPH_DTA_BASE}/davis/proteins.txt",
            "Y": f"{GRAPH_DTA_BASE}/davis/Y",
            "folds/train_fold_setting1.txt": f"{GRAPH_DTA_BASE}/davis/folds/train_fold_setting1.txt",
            "folds/test_fold_setting1.txt": f"{GRAPH_DTA_BASE}/davis/folds/test_fold_setting1.txt",
        },
        description="Davis kinase inhibitor affinity benchmark.",
    ),
    "kiba": BenchmarkSpec(
        name="kiba",
        raw_files={
            "ligands_can.txt": f"{GRAPH_DTA_BASE}/kiba/ligands_can.txt",
            "proteins.txt": f"{GRAPH_DTA_BASE}/kiba/proteins.txt",
            "Y": f"{GRAPH_DTA_BASE}/kiba/Y",
            "folds/train_fold_setting1.txt": f"{GRAPH_DTA_BASE}/kiba/folds/train_fold_setting1.txt",
            "folds/test_fold_setting1.txt": f"{GRAPH_DTA_BASE}/kiba/folds/test_fold_setting1.txt",
        },
        description="KIBA kinase inhibitor benchmark.",
    ),
    "pdbbind_v2016": BenchmarkSpec(
        name="pdbbind_v2016",
        raw_files={
            "LP_PDBBind.csv": "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/LP_PDBBind.csv",
        },
        description=(
            "PDBbind v2016-like subset reconstructed from open LP-PDBBind metadata by filtering "
            "refined entries with deposition date <= 2016-12-31."
        ),
    ),
    "core2016": BenchmarkSpec(
        name="core2016",
        raw_files={
            "affinity_data.csv": f"{DATADTA_BASE}/affinity_data.csv",
            "training_smi.csv": f"{DATADTA_BASE}/training_smi.csv",
            "training_seq_.csv": f"{DATADTA_BASE}/training_seq_.csv",
            "validation_smi.csv": f"{DATADTA_BASE}/validation_smi.csv",
            "validation_seq_.csv": f"{DATADTA_BASE}/validation_seq_.csv",
            "test_smi.csv": f"{DATADTA_BASE}/test_smi.csv",
            "test_seq_.csv": f"{DATADTA_BASE}/test_seq_.csv",
        },
        description="PDBbind Core2016 split packaged in the DataDTA benchmark release.",
    ),
    "test71": BenchmarkSpec(
        name="test71",
        raw_files={
            "affinity_data.csv": f"{DATADTA_BASE}/affinity_data.csv",
            "training_smi.csv": f"{DATADTA_BASE}/training_smi.csv",
            "training_seq_.csv": f"{DATADTA_BASE}/training_seq_.csv",
            "validation_smi.csv": f"{DATADTA_BASE}/validation_smi.csv",
            "validation_seq_.csv": f"{DATADTA_BASE}/validation_seq_.csv",
            "test71_smi.csv": f"{DATADTA_BASE}/test71_smi.csv",
            "test71_seq_.csv": f"{DATADTA_BASE}/test71_seq_.csv",
        },
        description="PDBbind Test71 split packaged in the DataDTA benchmark release.",
    ),
    "test105": BenchmarkSpec(
        name="test105",
        raw_files={
            "affinity_data.csv": f"{DATADTA_BASE}/affinity_data.csv",
            "training_smi.csv": f"{DATADTA_BASE}/training_smi.csv",
            "training_seq_.csv": f"{DATADTA_BASE}/training_seq_.csv",
            "validation_smi.csv": f"{DATADTA_BASE}/validation_smi.csv",
            "validation_seq_.csv": f"{DATADTA_BASE}/validation_seq_.csv",
            "test105_smi.csv": f"{DATADTA_BASE}/test105_smi.csv",
            "test105_seq_.csv": f"{DATADTA_BASE}/test105_seq_.csv",
        },
        description="PDBbind Test105 split packaged in the DataDTA benchmark release.",
    ),
}
