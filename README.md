# KGCL-DTA / GCDTA Project

This repository contains a PyTorch implementation of a GCDTA-style drug-target affinity pipeline and its innovation extensions for the KGCL-DTA thesis work.

## Core model

- Drug encoder: RDKit molecular graph + 8-head GAT
- Target encoder: FASTA token embedding + physicochemical features + dilated CNN
- Interaction fusion: cross-attention between drug atoms and protein residues
- Contrastive branch: heterogeneous graph contrastive learning with InfoNCE
- Regression head: affinity prediction on the pKd/pKi scale

## Innovation modules

- `pocket_uncertainty`
- `multitask_pose`
- `knowledge_graph`
- `structural_negatives`

## Repository structure

```text
.
|-- configs/
|-- data/
|   |-- processed/
|   `-- raw/
|-- paper_assets/
|   |-- data/
|   `-- templates/
|-- results/
|-- scripts/
|-- src/gcdta/
|-- evaluate.py
|-- evaluate_innovations.py
|-- predict.py
|-- predict_innovations.py
|-- train.py
|-- train_innovations.py
`-- requirements.txt
```

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For paper figures, install R and the packages used by `scripts/generate_paper_figures.R`:

```r
install.packages(c("ggplot2", "patchwork", "readr", "dplyr", "tidyr"))
```

## Dataset preparation

Download and preprocess all supported datasets into `data/`:

```bash
python scripts/prepare_data.py --dataset all
```

Supported datasets:

- `davis`
- `kiba`
- `core2016`
- `test71`
- `test105`
- `pdbbind_v2016`

## Base training

Train the baseline GCDTA model:

```bash
python train.py --dataset davis
```

## Innovation training

Smoke-test style training:

```bash
python train_innovations.py --config configs/base.yaml --epochs 3
python train_innovations.py --config configs/path1_pocket_uncertainty.yaml --epochs 3
python train_innovations.py --config configs/path2_multitask_pose.yaml --epochs 3
python train_innovations.py --config configs/path3_knowledge_graph.yaml --epochs 3
python train_innovations.py --config configs/path4_structural_negatives.yaml --epochs 3
```

Final Davis experiment configs:

```bash
python train_innovations.py --config configs/base_final.yaml
python train_innovations.py --config configs/path1_final.yaml
python train_innovations.py --config configs/path2_final.yaml
python train_innovations.py --config configs/path3_final.yaml
python train_innovations.py --config configs/path4_final.yaml
```

Final KIBA experiment configs:

```bash
python train_innovations.py --config configs/base_kiba_final.yaml
python train_innovations.py --config configs/path1_kiba_final.yaml
python train_innovations.py --config configs/path2_kiba_final.yaml
python train_innovations.py --config configs/path3_kiba_final.yaml
python train_innovations.py --config configs/path4_kiba_final.yaml
```

Batch runners:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_experiments.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_kiba_final_experiments.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_ablation_experiments.ps1
```

## Ablation runs

Ready-made ablation configs:

```bash
python train_innovations.py --config configs/ablation_no_cross_attention.yaml
python train_innovations.py --config configs/ablation_no_contrastive.yaml
python train_innovations.py --config configs/ablation_no_physchem.yaml
python train_innovations.py --config configs/ablation_no_kg.yaml
```

Supported ablation toggles in config:

```yaml
ablations:
  disable_cross_attention: false
  disable_contrastive: false
  disable_physchem: false
  disable_knowledge_graph: false
```

## Evaluation

Base model:

```bash
python evaluate.py --dataset davis --model-path results/best_model.pth
```

Innovation comparison:

```bash
python evaluate_innovations.py --dataset davis --model-path results/best_model.pth --compare-all
```

## Prediction

Base single-pair prediction:

```bash
python predict.py --model-path results/best_model.pth --smiles "CCO" --fasta "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
```

Innovation comparison on a single pair:

```bash
python predict_innovations.py --model-path results/best_model.pth --smiles "CCO" --fasta "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE" --compare-all
```

## What each final run saves

Each final or ablation result folder now contains paper-usable outputs such as:

- `best_model.pth`
- `config.json`
- `logs.csv`
- `training_summary.json`
- `validation_predictions.csv`
- `test_predictions.csv`
- `predictions_scatter.png`
- `training_curves.png`
- `attention_matrix.csv` when `save_attention_matrix: true`

These CSV exports are intended for paper plots, heatmaps, scatter figures, and comparison tables.

## Metrics aggregation

After final runs complete, aggregate validation and test metrics:

```bash
python scripts/compare_final_runs.py
```

This writes:

- `paper_assets/data/final_runs_metrics.csv`

## Paper figure generation in R

Dataset distributions:

```bash
Rscript scripts/generate_paper_figures.R distributions
```

Training curves:

```bash
Rscript scripts/generate_paper_figures.R training_curves
```

Scatter plots from exported run CSVs:

```bash
Rscript scripts/generate_paper_figures.R scatter_from_runs
```

Performance bar plots:

```bash
Rscript scripts/generate_paper_figures.R performance
```

Ablation plots:

```bash
Rscript scripts/generate_paper_figures.R ablation
```

Cross-attention heatmap from exported attention matrix:

```bash
Rscript scripts/generate_paper_figures.R attention_from_run
```

Metrics heatmap from aggregated final runs:

```bash
Rscript scripts/generate_paper_figures.R metrics_heatmap
```

KG case-study figure:

```bash
Rscript scripts/generate_paper_figures.R kg_case
```

All generated paper figures are written to:

- `results/paper_figures/`

## Notes

- The earlier 3-epoch runs are smoke tests only. They should not be used as final paper results.
- The distribution plots are dataset-based and do not change when you retrain.
- Scatter plots, performance bars, ablation plots, training curves, and heatmaps depend on final experiment outputs.
- Existing old checkpoints may be incompatible with the current model definition. Retraining is required for stable evaluation.
