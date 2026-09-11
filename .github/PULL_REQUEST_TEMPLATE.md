## What does this PR do?

<!-- Bug fix, doc update, new model candidate, preprocessing change, etc. -->

## If this adds or changes a model

- [ ] Followed the evaluation protocol in `docs/reproducing-training.md` §4
      (cluster-aware split, macro-F1, same held-out test set)
- [ ] Added/updated a row in `metadata/model_comparison.csv` (or a new CSV, documented)
- [ ] Did **not** attempt to push checkpoints to `models/checkpoints.dvc` / the private
      R2 remote -- linked checkpoints externally instead if reviewers need them
- [ ] Linked the related issue (if this follows a Model Proposal issue)

## If this changes data/preprocessing

- [ ] Re-ran `scripts/generate_manifest.py` / `scripts/preprocess_dataset.py` and
      confirmed downstream manifests still match the expected schema
- [ ] Noted any change to `docs/preprocessing-recommendations.md` if the rationale changed

## Checklist

- [ ] Docs updated if behavior/results changed
- [ ] No secrets or credentials included (check `git diff` before pushing)
