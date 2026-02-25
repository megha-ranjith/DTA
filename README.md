# GCDTA (Graph-Attention-Assisted Contrastive Learning for Drug-Target Affinity)

This project implements a GCDTA-style drug-target affinity framework in PyTorch with:

- **Drug encoder**: RDKit molecular graph + **8-head GAT**.
- **Target encoder**: FASTA residue features + **multi-layer dilated CNN (PReLU)**.
- **Fusion**: **cross-attention** between drug and target token embeddings.
- **Contrastive branch**: **HGCN + InfoNCE** over batch-level heterogeneous drug-target graph.
- **Prediction head**: regression to affinity score (pKd/pKi scale).

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Download + preprocess datasets into `./data`

```bash
python scripts/prepare_data.py --dataset all
```

Supported datasets:

- `davis`
- `kiba`
- `pdbbind_v2016`

Raw files are downloaded to `data/raw/<dataset>/` and standardized CSV files are written to `data/processed/<dataset>.csv`.

### Notes on PDBbind v2016

`pdbbind_v2016` is prepared from an open-source PDBbind metadata table by filtering **refined** entries with deposition date `<= 2016-12-31`, then creating train/val/test splits.

## 3) Train

```bash
python train.py --dataset davis
```

Training artifacts are saved automatically:

- `results/best_model.pth`
- `results/logs.csv`
- `results/scatter_plot.png`

## 4) Evaluate

```bash
python evaluate.py --dataset davis --model-path results/best_model.pth
```

This prints:

- CI
- MSE
- Pearson R
- RMSE

And writes `results/<dataset>_performance.txt`.

## 5) Predict (single pair)

```bash
python predict.py --model-path results/best_model.pth --smiles "CCO" --fasta "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
```

Output format:

```text
Predicting affinity for Drug-Target pair...
Drug SMILES: [Input SMILES]
Target FASTA: [Input FASTA]
--------------------------------------------------
Predicted Affinity Score (pKd/pKi): [Numerical Value]
Processing Time: [X]s
```

## Project layout

```text
.
├─ data/
├─ results/
├─ scripts/
│  └─ prepare_data.py
├─ src/gcdta/
│  ├─ data/
│  ├─ models/
│  ├─ losses.py
│  ├─ metrics.py
│  ├─ runtime.py
│  └─ train_utils.py
├─ train.py
├─ evaluate.py
├─ predict.py
└─ requirements.txt
```

