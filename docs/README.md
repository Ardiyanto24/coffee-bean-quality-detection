# Documentation Index

Start at the main [README](../README.md) for the project overview, or
[`MODEL_CARD.md`](../MODEL_CARD.md) for the final model's full spec. Everything below is
deeper reference material.

## In English (project-wide reference)

| Doc | Covers |
|---|---|
| [`data.md`](data.md) | Dataset source (Kaggle), what's DVC-tracked vs. not, how to get data without maintainer access |
| [`reproducing-training.md`](reproducing-training.md) | Environment setup, notebook execution order, evaluation protocol for new models |
| [`results.md`](results.md) | Full 10-model comparison, HPO, 4-fold CV, seed-stability evidence, final decision |
| [`architecture.md`](architecture.md) | The 9-phase project pipeline (data -> experiments -> production model) |

## In Bahasa Indonesia (original strategy documents, written pre-implementation)

These were written during the project to plan each stage *before* implementing it, based
on EDA findings. They're kept as-is (original language) since they're referenced
throughout the codebase and reports; the English docs above summarize their outcomes for
readers who don't need the full original reasoning.

| Doc | Covers |
|---|---|
| [`modeling-strategy.md`](modeling-strategy.md) | The 10-model roadmap and why each was chosen, evaluation strategy, phased experiment plan |
| [`preprocessing-recommendations.md`](preprocessing-recommendations.md) | Deduplication, cluster-aware split, label-noise handling, augmentation guidance |
| [`xai-strategy.md`](xai-strategy.md) | Why each XAI method was chosen per model architecture family, the 5 shortcut-learning hypotheses |

## HTML Reports

Full interactive/narrative reports live in [`../reports/`](../reports/) -- indexed in
[`architecture.md`](architecture.md#reports-index).
