# KGCL-DTA / GCDTA Project

KGCL-DTA is a modular drug-target affinity (DTA) prediction project built around a GCDTA-style graph-sequence backbone and extended with four research paths:

- Path 1: uncertainty-aware pocket modeling
- Path 2: joint pose-and-affinity style modeling
- Path 3: knowledge-guided novelty support
- Path 4: structural hard-negative learning

The project predicts a continuous binding-affinity value from a drug SMILES string and a target protein FASTA sequence.

The current implementation is a working DTA research framework. It implements the architecture and forward logic for all four innovation paths, while the full ESMFold, real docking-pose supervision, curated PrimeKG integration, and DecoyDB-scale pretraining pipelines remain future extensions.

## Core model

- Drug encoder: RDKit molecular graph + edge-aware GATv2
- Target encoder: FASTA token embedding + physicochemical residue features + dilated CNN
- Interaction fusion: cross-attention between drug atom tokens and protein residue tokens
- Contrastive branch: heterogeneous graph contrastive learning with InfoNCE
- Regression head: continuous affinity prediction
- KG support: Morgan fingerprint retrieval + KG-style embeddings + mock PrimeKG-style relation demo

## Current verified results

The current final-run story is tradeoff-based, not a single universal winner.

| Dataset | Best error behavior | Best ranking/correlation behavior |
|---|---|---|
| Davis | Path 1 gives best MSE/RMSE/Pearson: MSE 0.4726, RMSE 0.6874, Pearson 0.6504 | Path 3 gives best CI: 0.8091 |
| KIBA | Base gives best MSE/RMSE: MSE 0.3967, RMSE 0.6299 | Path 1 gives best CI/Pearson: CI 0.7563, Pearson 0.6547 |

Full verified test-set comparison:

| Variant | Davis CI | Davis MSE | Davis RMSE | Davis Pearson | KIBA CI | KIBA MSE | KIBA RMSE | KIBA Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base backbone | 0.8034 | 0.4887 | 0.6991 | 0.6267 | 0.7479 | 0.3967 | 0.6299 | 0.6455 |
| Path 1: Pocket uncertainty | 0.8087 | 0.4726 | 0.6874 | 0.6504 | 0.7563 | 0.4203 | 0.6483 | 0.6547 |
| Path 2: Multitask pose | 0.8039 | 0.4842 | 0.6958 | 0.6324 | 0.7493 | 0.4088 | 0.6394 | 0.6450 |
| Path 3: Knowledge graph | 0.8091 | 0.4868 | 0.6977 | 0.6369 | 0.7477 | 0.4308 | 0.6564 | 0.6302 |
| Path 4: Structural negatives | 0.8058 | 0.4895 | 0.6996 | 0.6303 | 0.7458 | 0.4370 | 0.6611 | 0.6319 |

Do not use older 3-epoch or smoke-test metrics as thesis results.

## Installation

```powershell
pip install -r requirements.txt
```

For paper figures, install R and packages used by `scripts/generate_paper_figures.R`.

## Dataset preparation

```powershell
python scripts\prepare_data.py --dataset all
```

Supported dataset names include `davis`, `kiba`, `core2016`, `test71`, `test105`, and `pdbbind_v2016`.

Core 2016/PDBbind support is available for future structure-aware benchmarking, but current thesis metrics are Davis/KIBA final runs.

## Training

Final Davis runs:

```powershell
python train_innovations.py --config configs\base_final.yaml
python train_innovations.py --config configs\path1_final.yaml
python train_innovations.py --config configs\path2_final.yaml
python train_innovations.py --config configs\path3_final.yaml
python train_innovations.py --config configs\path4_final.yaml
```

Final KIBA runs:

```powershell
python train_innovations.py --config configs\base_kiba_final.yaml
python train_innovations.py --config configs\path1_kiba_final.yaml
python train_innovations.py --config configs\path2_kiba_final.yaml
python train_innovations.py --config configs\path3_kiba_final.yaml
python train_innovations.py --config configs\path4_kiba_final.yaml
```

Ablation configs are prepared, but missing ablation values should not be fabricated:

```powershell
python train_innovations.py --config configs\ablation_no_cross_attention.yaml
python train_innovations.py --config configs\ablation_no_contrastive.yaml
python train_innovations.py --config configs\ablation_no_physchem.yaml
python train_innovations.py --config configs\ablation_no_kg.yaml
```

## Prediction and evaluation

Base prediction:

```powershell
python predict.py --model-path results\base_final\best_model.pth --smiles "CCO" --fasta "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
```

Innovation prediction:

```powershell
python predict_innovations.py --model-path results\path3_final\best_model.pth --config configs\path3_knowledge_graph.yaml --smiles "CCO" --fasta "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE" --dataset davis
```

Dataset-level evaluation:

```powershell
python evaluate_innovations.py --model-path results\path3_final\best_model.pth --config configs\path3_knowledge_graph.yaml --dataset davis --output-dir results\demo_eval_path3
```

## Demo tools

Local molecule and pipeline viewer:

```powershell
python smiles_3d_gui.py
```

This shows SMILES-to-2D/3D RDKit visualization, molecule statistics, drug graph construction, protein feature analysis, and the KGCL-DTA pipeline. The 3D molecule is an RDKit-generated conformer for visualization, not an experimentally solved docking pose.

Optional web-style demo:

```powershell
streamlit run streamlit_app.py
```

Use it for viva convenience only; benchmark claims should come from saved Davis/KIBA evaluations.

## Unseen-split and mock-KG support

Create scaffold/protein-cluster split files:

```powershell
python scripts\create_unseen_splits.py --dataset davis --mode both
python scripts\create_unseen_splits.py --dataset kiba --mode both
```

Build mock PrimeKG-style graph files:

```powershell
python scripts\build_mock_primekg.py --dataset davis
python scripts\build_mock_primekg.py --dataset kiba
```

These are support tools. They should be described as prepared infrastructure unless full training/evaluation is performed on the generated splits or real PrimeKG data.

## Paper figures

```powershell
Rscript scripts\generate_paper_figures.R distributions
Rscript scripts\generate_paper_figures.R thesis_variant_comparison
Rscript scripts\generate_paper_figures.R thesis_training_curves
Rscript scripts\generate_paper_figures.R thesis_attention_panels
Rscript scripts\generate_paper_figures.R metrics_heatmap
```

Generated figures are written to `results/paper_figures/`.

## Thesis-safe framing

Use this wording:

> KGCL-DTA implements a multimodal graph-sequence DTA backbone with four architectural innovation paths: uncertainty-aware pocket modeling, multitask pose-style prediction, knowledge-guided novelty support, and structural hard-negative learning. Current Davis/KIBA results show tradeoffs across metrics rather than one universally dominant variant. Full ESMFold, real PrimeKG, pose-supervised, DecoyDB, and multi-seed studies remain future work.
