---
name: Model proposal
about: Propose a model that beats (or meaningfully challenges) the current production baseline
title: "[Model Proposal] "
labels: model-proposal
---

<!--
Before filling this in, please read:
- docs/reproducing-training.md §4 (mandatory evaluation protocol)
- docs/results.md (the table your numbers should be comparable to)
-->

## Architecture

<!-- Backbone, key layers, params (roughly), anything non-standard. -->

## Training recipe

<!-- Loss function, augmentation, hyperparameters, how many seeds/runs. -->

## Evaluation protocol confirmation

- [ ] Evaluated on the same held-out test split produced by `scripts/preprocess_dataset.py` (not a custom split)
- [ ] Used cluster-aware CV (`fold` / `cluster_id` columns) if you did any cross-validation
- [ ] Reporting macro-F1 (not just accuracy)
- [ ] Ran more than one seed, if claiming a specific score (single-run numbers on this
      project have previously been found not to reproduce -- see `docs/results.md` §4)

## Results

| Metric | Value |
|---|---|
| Test macro-F1 | |
| Test accuracy | |
| Recall `defect` | |
| Recall `longberry` | |
| Recall `peaberry` | |
| Recall `premium` | |
| Number of seeds/runs | |

## Comparison to current baseline

<!-- How does this compare to 09_noise_robust (0.9611 test macro-F1)? -->

## Code

<!-- Link to your notebook/script/branch, or attach it if not yet in a PR. -->
