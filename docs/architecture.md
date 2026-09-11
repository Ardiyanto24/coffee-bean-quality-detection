# Project Architecture -- The 9-Phase Pipeline

This document describes the *project's* pipeline (data -> experiments -> production
model), end to end. For the *final model's* own internal architecture (backbone, loss,
ensembling), see [`MODEL_CARD.md`](../MODEL_CARD.md) instead -- these are two different
scopes and deliberately kept in separate files.

Interactive version: [`reports/CBQD - ML Pipeline Architecture (9 Phases).html`](../reports/CBQD%20-%20ML%20Pipeline%20Architecture%20(9%20Phases).html).

![The 9-phase pipeline as a snake-flow diagram: Data Source, EDA, Generate Manifest, Preprocess Dataset, Model Training, HPO Tuning, Validation, Explainability, Final Production Model, color-coded by category (data, data prep, experimentation, validation, production artifact).](images/pipeline-9-phase.png)

## Pipeline Overview

```
01 Data Source            Raw dataset, DVC-tracked, 4 classes (public Kaggle origin)
        |
02 EDA (3 rounds)         class & label-noise analysis -> docs/preprocessing-recommendations.md, docs/modeling-strategy.md
        |
03 Generate Manifest      scripts/generate_manifest.py -> metadata/manifest.csv
        |
04 Preprocess Dataset     scripts/preprocess_dataset.py: dedup, cluster-aware split, crop+resize
        |                 -> dataset_preprocessed/, metadata/manifest_preprocessed.csv
        |                 (dataset.dvc -> Cloudflare R2 remote)
        |
05 Model Training         10-model bake-off: CNN, ViT, tabular, hybrid architectures
   (Bake-off)              -> 7 eliminated below top-3 macro-F1, 3 advance
        |
06 HPO Tuning              Optuna, top-3 candidates only
        |
07 Validation               4-fold CV (cluster-aware) + repeated-seed stability reruns
   (top-3 HPO candidates)   -> reveals HPO's headline gain does not reproduce across seeds
        |
08 Explainability (XAI)    5 hypotheses x 5 methods, run across ALL 10 original models
        |                  -> verifies ranking reflects real signal, not shortcut learning
        |
09 Final Production Model  09_noise_robust: 5-seed ensemble
                            (models/checkpoints.dvc -> Cloudflare R2 remote)
```

## Phase Details

| Phase | Artifact(s) | Notebook(s) / Script(s) |
|---|---|---|
| 01. Data Source | `dataset/` (DVC) | -- (external, from Kaggle; see [`docs/data.md`](data.md)) |
| 02. EDA | `reports/*EDA*.html` | `CBQD - EDA.ipynb`, `CBQD - EDA v2 (Manual).ipynb`, `CBQD - EDA v2 (FiftyOne).ipynb`, `CBQD - EDA v3 (Defect Decomposition).ipynb` |
| 03. Generate Manifest | `metadata/manifest.csv` | `scripts/generate_manifest.py` |
| 04. Preprocess Dataset | `dataset_preprocessed/`, `metadata/manifest_preprocessed.csv` | `scripts/preprocess_dataset.py`, exploratory version in `CBQD - Preprocessing.ipynb` |
| 05. Model Training | `metadata/model_comparison.csv` | `CBQD - Training.ipynb` |
| 06. HPO Tuning | `metadata/hpo_final_summary.csv`, `results/hpo_trials_*.csv` | `CBQD - HPO Optuna.ipynb` |
| 07. Validation | `metadata/cv_4fold_*.csv`, `metadata/hpo_seed_stability_*.csv` | `CBQD - 4Fold CV.ipynb`, `CBQD - HPO Seed Stability.ipynb` |
| 08. Explainability | `metadata/xai_*.csv` | `CBQD - XAI.ipynb`, `CBQD - XAI HPO-Tuned.ipynb`, `CBQD - XAI 09-Noise-Robust Full.ipynb` |
| 09. Final Production Model | `models/checkpoints/09_noise_robust_ensemble_seed*.pt` | `CBQD - 09 Noise-Robust Ensemble.ipynb` |

## What's automated (DVC) vs. manual (notebooks)

Only phases 03-04 are wired as a formal [DVC pipeline](../dvc.yaml) (`generate_manifest`
-> `preprocess_dataset`), meaning they're reproducible with a single `dvc repro` and
their outputs are content-addressed and cached. Phases 05-09 (training, HPO, CV, XAI,
final ensembling) are run as notebooks against the phase-04 output and are **not**
currently DVC pipeline stages -- their results are versioned as plain CSVs in `metadata/`
(readable in git diffs) and their model weights are pushed to DVC/R2 manually at the end
of each notebook. `docs/preprocessing-recommendations.md` §7 has a recommendation to
formalize more of this as DVC stages (`validate_dataset`, `clean_manifest`,
`assign_folds`) -- not yet implemented.

## Reports Index

| Report | Covers |
|---|---|
| [`CBQD - EDA v2 Report.html`](../reports/CBQD%20-%20EDA%20v2%20Report.html) | Manual EDA: duplicates, class balance, color/texture/shape signal |
| [`CBQD - EDA v3 (Defect Decomposition) Report.html`](../reports/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition)%20Report.html) | Hypothesis that `defect` is a cross-cutting condition, not an independent class |
| [`CBQD - Training Report.html`](../reports/CBQD%20-%20Training%20Report.html) | Full 10-model bake-off writeup |
| [`CBQD - HPO Report.html`](../reports/CBQD%20-%20HPO%20Report.html) | Optuna tuning results and the ensemble failure investigation |
| [`CBQD - XAI Report.html`](../reports/CBQD%20-%20XAI%20Report.html) | Full interpretability findings across all 10 models |
| [`CBQD - Model Selection Report.html`](../reports/CBQD%20-%20Model%20Selection%20Report.html) | The complete 9-phase narrative, from EDA to final ensemble |
| [`CBQD - ML Pipeline Architecture (9 Phases).html`](../reports/CBQD%20-%20ML%20Pipeline%20Architecture%20(9%20Phases).html) | Interactive version of this document |
| [`CBQD - Model Architecture Diagram.html`](../reports/CBQD%20-%20Model%20Architecture%20Diagram.html) | Interactive version of `MODEL_CARD.md` §3 |
