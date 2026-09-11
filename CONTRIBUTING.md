# Contributing

Contributions are welcome, especially: retraining on your own environment to verify the
results, proposing a model that beats the current production baseline, improving
preprocessing/label-quality handling, or extending the XAI analysis.

## Ways to contribute

- **Reproduce the results** and report back if you get materially different numbers --
  useful signal on its own, see [`docs/results.md`](docs/results.md) for what to expect.
- **Propose a better model.** The current production model (`09_noise_robust`, a 5-seed
  EfficientNet-B0 ensemble, test macro-F1 0.9611) is documented in full in
  [`MODEL_CARD.md`](MODEL_CARD.md). If you can beat it under the same evaluation
  protocol, open a PR (see below).
- **Improve data/label quality.** [`docs/preprocessing-recommendations.md`](docs/preprocessing-recommendations.md)
  lists several open manual-review items (mislabel candidates, ambiguous cross-class
  pairs) that were intentionally left for human review rather than automated.
- **Fix bugs / improve docs.** Regular PRs, no special process needed.

## Before you start training

1. Set up the environment: see [`docs/reproducing-training.md` §1](docs/reproducing-training.md#1-environment).
2. Get the dataset from Kaggle (not this project's private R2 remote): see
   [`docs/data.md`](docs/data.md).
3. Read the mandatory evaluation protocol: [`docs/reproducing-training.md` §4](docs/reproducing-training.md#4-evaluation-protocol-read-before-training-your-own-model).
   A model evaluated with a plain random split, or reported as accuracy instead of
   macro-F1, is **not comparable** to the results in this repo and will be asked to be
   re-evaluated before review.

## Submitting a model proposal

1. Train and evaluate your model following the protocol above, on the same held-out
   test split produced by `scripts/preprocess_dataset.py` (do not create your own split).
2. Open an issue using the **Model Proposal** template
   (`.github/ISSUE_TEMPLATE/model_proposal.md`) with your architecture, hyperparameters,
   and full metrics (macro-F1, accuracy, per-class recall at minimum).
3. If it looks promising, follow up with a PR that adds:
   - Your notebook or training script under `notebook/` or `scripts/`.
   - A new row in `metadata/model_comparison.csv` (or a new CSV if the shape doesn't fit)
     with your results, matching the existing column schema.
   - A short write-up of your approach (a new file under `docs/` or a section in your
     PR description is fine -- doesn't need to match the depth of the existing docs).
4. Do **not** attempt to push checkpoints to `models/checkpoints.dvc` / the R2 remote --
   that remote is private to the maintainer. Share checkpoints via the PR/issue (e.g. a
   release asset or external link) if reviewers need to verify them; the maintainer will
   fold accepted checkpoints into DVC-tracked storage separately.

## Reporting issues

Use a regular GitHub issue for bugs, unclear docs, or reproducibility problems. Include:
- What you ran (which notebook/script, what step).
- What you expected vs. what happened.
- Your environment (OS, Python version, GPU or CPU).

## Code style

This is primarily a notebook-driven research project. There's no enforced linter/style
config yet -- keep new notebooks structured similarly to existing ones (clear
sub-step markdown headers, DRY_RUN toggle for expensive cells where applicable) and keep
`scripts/*.py` dependency-light and documented with a module docstring, matching
[`scripts/preprocess_dataset.py`](scripts/preprocess_dataset.py) as the reference example.

## License

By contributing, you agree your contribution is licensed under this repository's
[MIT License](LICENSE).
