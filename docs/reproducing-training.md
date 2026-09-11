# Reproducing Training

Step-by-step to go from a fresh clone to a retrained model, without needing access to
this project's private DVC/R2 remote.

## 1. Environment

```bash
git clone https://github.com/Ardiyanto24/coffee-bean-quality-detection.git
cd coffee-bean-quality-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. The original notebooks were built and run on
Colab/Kaggle-style hosted GPU runtimes (each notebook's first cell runs its own
`!pip install -q ...` for that notebook's extra dependencies -- `requirements.txt`
consolidates all of them for a local setup). Training on CPU is possible but slow,
especially for the 10-model bake-off and 5-seed ensemble.

## 2. Get the data

Follow [`docs/data.md`](data.md): download the raw dataset from Kaggle
(`coffee-bean-classification-dataset`), lay it out as `dataset/train/<class>/*.jpg` +
`dataset/test/*.jpg`, then run:

```bash
python scripts/generate_manifest.py
python scripts/preprocess_dataset.py
```

This produces `metadata/manifest.csv`, `dataset_preprocessed/`, and
`metadata/manifest_preprocessed.csv` -- everything downstream reads from
`dataset_preprocessed/` and the preprocessed manifest, not the raw folder.

## 3. Notebook execution order

Notebooks live in [`notebook/`](../notebook/) and are meant to be run in this order.
Each one reads CSV/checkpoint outputs from the previous ones via `metadata/` and
`models/checkpoints/` -- skipping a step will make a later notebook fail to find its
inputs.

| Step | Notebook | Depends on | Produces |
|---|---|---|---|
| 1 | `CBQD - EDA.ipynb` | raw dataset | initial exploration |
| 2 | `CBQD - EDA v2 (Manual).ipynb` | raw dataset | `reports/CBQD - EDA v2 Report.html` source, hand-crafted feature baseline |
| 2b | `CBQD - EDA v2 (FiftyOne).ipynb` | raw dataset | optional alternate EDA tooling (needs `fiftyone`) |
| 3 | `CBQD - EDA v3 (Defect Decomposition).ipynb` | EDA v2 outputs | `defect`-as-mixture hypothesis |
| 4 | `CBQD - Preprocessing.ipynb` | raw dataset | exploratory reference for `scripts/preprocess_dataset.py` (already run in Section 2 above) |
| 5 | `CBQD - Training.ipynb` | `dataset_preprocessed/` | `metadata/model_comparison.csv`, 10 model checkpoints -- **the full bake-off, this is the long one** |
| 6 | `CBQD - HPO Optuna.ipynb` | Step 5 results | `metadata/hpo_final_summary.csv`, `results/hpo_trials_*.csv` -- tunes only the top-3 from Step 5 |
| 7 | `CBQD - 4Fold CV.ipynb` | Step 6 results | `metadata/cv_4fold_*.csv` |
| 8 | `CBQD - HPO Seed Stability.ipynb` | Step 6 results | `metadata/hpo_seed_stability_*.csv` |
| 9 | `CBQD - XAI.ipynb` | Step 5 results | `metadata/xai_*.csv` -- runs all 5 XAI methods across all 10 original models |
| 10 | `CBQD - XAI HPO-Tuned.ipynb` | Step 6 results | `metadata/xai_hpo_tuned_*.csv` -- rescoped to the 3 HPO-tuned candidates |
| 11 | `CBQD - XAI 09-Noise-Robust Full.ipynb` | Step 6/8 results | `metadata/xai_09_noise_robust_full_summary.csv` -- deep dive on the eventual winner |
| 12 | `CBQD - 09 Noise-Robust Ensemble.ipynb` | Steps 6-11 | `models/checkpoints/09_noise_robust_ensemble_seed*.pt`, `metadata/09_noise_robust_ensemble_*.csv` -- **the final production artifact** |

If you only want to reproduce the final model (not the full comparative study), you
still need Steps 1-2 done to have `dataset_preprocessed/`, then can jump straight to
Step 12 -- but note it reads `metadata/hpo_final_summary.csv` (Step 6's output) for its
tuned hyperparameters, so Step 6 for `09_noise_robust` specifically is a hard
dependency, not optional.

## 4. Evaluation protocol (read before training your own model)

To keep any new model's score comparable to the table in
[`docs/results.md`](results.md), it must follow the same protocol used throughout this
project (defined in [`docs/modeling-strategy.md` §3](modeling-strategy.md#3-strategi-evaluasi)):

- **Cluster-aware CV/split only.** Use the `fold` column from the preprocessed manifest
  (`StratifiedGroupKFold` by `cluster_id`). A plain random split will leak near-duplicate
  beans between train and validation and produce an optimistic, incomparable score.
- **Report macro-F1, not just accuracy.** Class difficulty is uneven (`defect` is
  historically the hardest); accuracy alone can hide that.
- **Beat the baseline.** Any new model must beat `01_gradient_boosting`'s 0.744 macro-F1
  test score (and ideally approach or beat `09_noise_robust`'s 0.961) on the *same*
  held-out test split -- see [`docs/results.md`](results.md) for the full comparison
  table your result should be placed alongside.
- **Report per-class recall**, especially `defect` -- a single macro-F1 number can hide
  which of the two underlying sub-problems (damage detection vs. type identification,
  see [`docs/modeling-strategy.md` §1](modeling-strategy.md)) your model is actually
  weak on.

If you get a promising result, see [`CONTRIBUTING.md`](../CONTRIBUTING.md) for how to
submit it.

## 5. Where checkpoints and DVC fit in

If you *are* the project maintainer (have R2 credentials configured locally), the normal
DVC flow applies:

```bash
dvc pull          # pull dataset/ and models/checkpoints/ from R2
# ... run notebooks, notebooks call `dvc add` + `dvc push` for new checkpoints ...
git add models/checkpoints.dvc metadata/*.csv
git commit -m "..."
```

External contributors should regenerate `dataset_preprocessed/` locally (Section 2) and
train fresh checkpoints locally -- there is no need to push anything to this project's
R2 remote to reproduce or extend the work; open a PR instead (see
[`CONTRIBUTING.md`](../CONTRIBUTING.md)).
