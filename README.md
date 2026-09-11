# Coffee Bean Quality Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![DVC](https://img.shields.io/badge/data%20%26%20models-DVC-13ADC7)

Image classification of single coffee beans into 4 quality/type classes --
`defect`, `longberry`, `peaberry`, `premium` -- built as an end-to-end, evidence-driven
ML study: 3 rounds of EDA, a 10-model architecture bake-off, hyperparameter optimization,
cross-validation and repeated-seed stability checks, and a full explainability (XAI)
audit, ending in a 5-seed noise-robust deep ensemble as the production model.

The project deliberately documents not just the winning model, but *why* it won over 9
other candidates -- including a finding that a promising +3pp HPO gain on the runner-up
turned out not to reproduce across seeds. See [`docs/results.md`](docs/results.md) for
the full comparison.

## Final Model & Performance

**`09_noise_robust`** -- a 5-seed ensemble of EfficientNet-B0 classifiers, trained with
label smoothing and mislabel-aware sample weighting, combined by softmax averaging.

| Metric | Value |
|---|---|
| Test macro-F1 | **0.9611** |
| Test accuracy | 0.9610 |
| Recall (`defect` / `longberry` / `peaberry` / `premium`) | 0.932 / 1.000 / 0.966 / 0.947 |

This is a deliberately short summary -- full architecture diagram, training recipe,
hyperparameters, per-class results, and known limitations (e.g. a real-world
generalization confidence gap) are in **[`MODEL_CARD.md`](MODEL_CARD.md)**, the
dedicated reference for the production model.

## Repository Structure

```
notebook/     13 notebooks: EDA -> preprocessing -> training -> HPO -> validation -> XAI -> final ensemble
scripts/      generate_manifest.py, preprocess_dataset.py -- the 2 formal DVC pipeline stages
docs/         written strategy docs (data, reproducing, results, architecture; see docs/README.md)
reports/      9 rendered HTML reports (EDA, training, HPO, XAI, model selection, architecture diagrams)
metadata/     CSV results for every experiment stage (model comparison, HPO, CV, seed-stability, XAI)
results/      raw Optuna trial logs
dataset/      raw images, DVC-tracked (not in git; see docs/data.md)
models/       trained checkpoints, DVC-tracked (not in git; see docs/data.md)
dvc.yaml      the 2-stage DVC pipeline: generate_manifest -> preprocess_dataset
```

## Quickstart

```bash
git clone https://github.com/Ardiyanto24/coffee-bean-quality-detection.git
cd coffee-bean-quality-detection
pip install -r requirements.txt
```

Get the dataset (public Kaggle source, no special access needed) and run the 2-stage
prep pipeline:

```bash
python scripts/generate_manifest.py
python scripts/preprocess_dataset.py
```

Then either load the production checkpoints directly (inference snippet in
[`MODEL_CARD.md` §7](MODEL_CARD.md#7-how-to-use)) or retrain from scratch following
[`docs/reproducing-training.md`](docs/reproducing-training.md).

> This project's own dataset/checkpoint storage runs through a **private** DVC + R2
> remote (maintainer-only). `docs/data.md` explains how to reproduce everything without
> needing access to it.

## Documentation

| | |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Final model: architecture, training recipe, metrics, limitations |
| [`docs/results.md`](docs/results.md) | Full 10-model comparison, HPO, CV, seed-stability evidence |
| [`docs/data.md`](docs/data.md) | Dataset source, DVC layout, how to get the data |
| [`docs/reproducing-training.md`](docs/reproducing-training.md) | Environment, notebook order, evaluation protocol |
| [`docs/architecture.md`](docs/architecture.md) | The 9-phase project pipeline |
| [`docs/README.md`](docs/README.md) | Full documentation index (includes the original Bahasa Indonesia strategy docs) |

## Contributing

Trained a model that beats `09_noise_robust`? Found a bug in the pipeline? See
[`CONTRIBUTING.md`](CONTRIBUTING.md) -- it covers the mandatory evaluation protocol
(cluster-aware CV, macro-F1) and how to submit a model proposal.

## License

Code, notebooks, scripts, and docs are [MIT licensed](LICENSE). The dataset is sourced
from a third-party Kaggle dataset and is not covered by this license -- see
[`docs/data.md`](docs/data.md#1-source).
