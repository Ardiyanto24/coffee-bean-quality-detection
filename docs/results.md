# Results -- Full Model Comparison

This is the consolidated numeric record behind the model selection decision. For the
narrative version (why each step happened), see
[`reports/CBQD - Model Selection Report.html`](../reports/CBQD%20-%20Model%20Selection%20Report.html).
For the final model's own architecture and metrics in isolation, see
[`MODEL_CARD.md`](../MODEL_CARD.md).

All numbers below are read directly from `metadata/*.csv` (linked per section) -- nothing
here is recomputed or rounded beyond what the source CSV already contains.

## 1. Stage 1 -- 10-Model Bake-off

Source: [`metadata/model_comparison.csv`](../metadata/model_comparison.csv),
notebook [`CBQD - Training.ipynb`](../notebook/CBQD%20-%20Training.ipynb).

Every model here was evaluated on the same cluster-aware held-out test split (mandatory
protocol, see [`docs/modeling-strategy.md` §3](modeling-strategy.md#3-strategi-evaluasi)).
The gate: beat the 73% hand-crafted-feature RandomForest baseline.

| # | Model | Test macro-F1 | Test accuracy | Recall `defect` | Recall `longberry` | Recall `peaberry` | Recall `premium` | Beats baseline |
|---|---|---|---|---|---|---|---|---|
| 09 | noise_robust (single seed, pre-ensemble) | **0.9610** | 0.9610 | 0.9492 | 1.0000 | 0.9655 | 0.9298 | Yes |
| 05 | convnext_tiny | 0.9479 | 0.9481 | 0.9153 | 0.9474 | 0.9828 | 0.9474 | Yes |
| 08 | multitask | 0.9397 | 0.9394 | 0.8983 | 0.9649 | 0.9655 | 0.9298 | Yes |
| 06 | deit_tiny | 0.9345 | 0.9351 | 0.9492 | 0.9649 | 0.9655 | 0.8596 | Yes |
| 07 | hierarchical | 0.9268 | 0.9264 | 0.8983 | 0.9649 | 0.9483 | 0.8947 | Yes |
| 04 | resnet18 | 0.9261 | 0.9264 | 0.8983 | 0.9825 | 0.9483 | 0.8772 | Yes |
| 02 | mobilenet_v3_large | 0.9174 | 0.9177 | 0.8814 | 0.9825 | 0.9655 | 0.8421 | Yes |
| 03 | efficientnet_b0 | 0.9141 | 0.9134 | 0.8814 | 0.8947 | 0.9310 | 0.9474 | Yes |
| 10 | ensemble (01 + 04) | 0.8907 | 0.8918 | 0.7966 | 0.9825 | 0.9310 | 0.8596 | Yes |
| 01 | gradient_boosting (11 hand-crafted features) | 0.7442 | 0.7446 | 0.7288 | 0.8246 | 0.7241 | 0.7018 | Yes (is the baseline) |

**Observation flagged at this stage**: `10_ensemble` (gradient boosting + ResNet-18,
simple 50/50 probability average) scored *worse* than ResNet-18 alone (0.891 vs. 0.926).
This was left as an open question here and mechanistically explained later in XAI
(the two components disagree most exactly on the cases ResNet-18 gets right, so the
50/50 average pulls those predictions the wrong way -- see
[`docs/xai-strategy.md` §4E](xai-strategy.md)).

**Top-3 advanced to HPO**: `09_noise_robust`, `05_convnext_tiny`, `08_multitask`.

## 2. Stage 2 -- HPO (Optuna), Top-3 Only

Source: [`metadata/hpo_final_summary.csv`](../metadata/hpo_final_summary.csv),
notebook [`CBQD - HPO Optuna.ipynb`](../notebook/CBQD%20-%20HPO%20Optuna.ipynb).

| Model | Baseline test macro-F1 | Tuned test macro-F1 | Delta |
|---|---|---|---|
| 08_multitask | 0.9397 | **0.9698** | **+0.0301** |
| 05_convnext_tiny | 0.9479 | 0.9483 | +0.0004 |
| 09_noise_robust | 0.9610 | 0.9610 | -0.00002 |

At face value, HPO made `08_multitask` look like the new winner by a wide margin
(+3pp). This single number is exactly what Stage 3-4 exist to stress-test.

## 3. Stage 3 -- 4-Fold Cross-Validation

Source: [`metadata/cv_4fold_summary.csv`](../metadata/cv_4fold_summary.csv),
notebook [`CBQD - 4Fold CV.ipynb`](../notebook/CBQD%20-%204Fold%20CV.ipynb).

| Model | Test macro-F1 (mean of 4 folds) | Std across folds | Min | Max |
|---|---|---|---|---|
| **09_noise_robust** | **0.9470** | **0.0130** | 0.9305 | 0.9611 |
| 05_convnext_tiny | 0.9404 | 0.0189 | 0.9133 | 0.9566 |
| 08_multitask | 0.9347 | 0.0147 | 0.9172 | 0.9521 |

Under cluster-aware 4-fold CV, `09_noise_robust` is both the **highest-mean** and
**most fold-to-fold-stable** of the top-3 -- the opposite ranking from the single-run
HPO table above.

## 4. Stage 4 -- Repeated-Seed Stability (the deciding evidence)

Source: [`metadata/hpo_seed_stability_vs_references.csv`](../metadata/hpo_seed_stability_vs_references.csv),
notebook [`CBQD - HPO Seed Stability.ipynb`](../notebook/CBQD%20-%20HPO%20Seed%20Stability.ipynb).

This table lines up **every single-run number reported so far** for the top-3 next to a
5-seed repeated-rerun mean, to answer: was any of those single numbers a fluke?

| Model | 4-fold CV (pre-HPO) | HPO report (1 run) | XAI-verify (1 run) | **5-seed mean** | 5-seed std |
|---|---|---|---|---|---|
| 09_noise_robust | 0.9470 ± 0.0130 | 0.9610 | 0.9739 | **0.9540** | 0.0171 |
| 08_multitask | 0.9347 ± 0.0147 | 0.9698 | 0.9352 | **0.9532** | 0.0198 |
| 05_convnext_tiny | 0.9404 ± 0.0189 | 0.9483 | 0.9267 | **0.9448** | 0.0115 |

**This is the smoking gun for "HPO's +3pp gain is not reproducible."** `08_multitask`'s
single HPO run (0.9698) and its single XAI-verification rerun (0.9352) differ by 3.5pp
on the *same tuned hyperparameters* -- a swing entirely explained by random seed, not by
any real improvement. `09_noise_robust`'s single-run numbers (0.9610, 0.9739), by
contrast, stay consistently near its own 5-seed mean (0.9540). Once averaged over 5
seeds, `09_noise_robust` has both the highest mean macro-F1 of the three *and* is
free of the single-run volatility that made `08_multitask` briefly look best.

## 5. Final Decision -> Production Ensemble

Given Stage 3-4, `09_noise_robust` was confirmed as the strongest and most stable
candidate. Rather than ship the one lucky checkpoint from any single seed, the final
production artifact is a **5-seed deep ensemble** (softmax-averaged), which trades a
small amount of best-case peak score for materially better run-to-run reliability.
Full detail, architecture, and limitations: [`MODEL_CARD.md`](../MODEL_CARD.md).

| | Test macro-F1 |
|---|---|
| Mean of 5 individual seeds | 0.9540 ± 0.0171 |
| **5-seed ensemble (production)** | **0.9611** |

## 6. Explainability Cross-Check

Before trusting any of the numbers above as a fair model comparison, all 10 models
(not just the winner) were run through a 5-hypothesis x 5-method XAI protocol to check
*how* each model reaches its decisions, not just its score. No shortcut-learning red
flags were found for `09_noise_robust`. Full methodology and per-model findings:
[`docs/xai-strategy.md`](xai-strategy.md) and
[`reports/CBQD - XAI Report.html`](../reports/CBQD%20-%20XAI%20Report.html).

## 7. Want to beat this baseline?

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the evaluation protocol your candidate
model must follow (cluster-aware CV, macro-F1, the same held-out test split) before a
new number here is comparable to the ones in this table.
