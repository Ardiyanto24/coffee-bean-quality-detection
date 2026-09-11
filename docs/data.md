# Data

## 1. Source

The raw dataset originates from the public Kaggle dataset **`coffee-bean-classification-dataset`**
(referenced inside the original EDA notebook's raw path,
`kaggle/input/coffee-bean-classification-dataset/...`). It contains single-bean, plain-background
256x256 RGB images across 4 classes: `defect`, `longberry`, `peaberry`, `premium`.

**If you want to train your own model on this project's data, get it from Kaggle directly** --
search for `coffee-bean-classification-dataset` on Kaggle and download it. You do **not** need
access to this project's private DVC remote to reproduce training from raw data; see Section 3.

> The dataset itself is subject to its original Kaggle license/terms, separate from this
> repository's MIT license (code, notebooks, docs only) -- check the dataset page on Kaggle for
> current terms before redistributing it.

## 2. What's in this repo vs. what's DVC-tracked

| Path | Tracked by | Contents |
|---|---|---|
| `dataset/` | DVC (`dataset.dvc`) | Raw images as pulled from Kaggle, arranged as `train/<class>/*.jpg` + `test/*.jpg` |
| `dataset_preprocessed/` | Not committed (gitignored, regenerated locally) | Output of `scripts/preprocess_dataset.py` -- deduplicated, cluster-split, cropped+resized |
| `metadata/manifest.csv` | git (plain CSV) | Output of `scripts/generate_manifest.py` -- one row per raw image, path + label + split |
| `metadata/manifest_preprocessed.csv` | git (plain CSV) | Output of preprocessing, with full traceability back to `orig_path` |
| `models/checkpoints/` | DVC (`models/checkpoints.dvc`) | Trained model weights (~505MB across 18 files: all HPO/CV/seed-sweep/final-ensemble checkpoints) |

Both `dataset.dvc` and `models/checkpoints.dvc` point at a **private** Cloudflare R2 remote
(`r2remote`, configured in `.dvc/config`) that only the project maintainer can push/pull from.
External contributors will get a permission error running `dvc pull` against this remote --
that is expected, not a bug. Use Section 3 instead.

## 3. Getting the data without maintainer access

1. Download the raw dataset from Kaggle (`coffee-bean-classification-dataset`).
2. Arrange it locally to match the expected layout:
   ```
   dataset/
     train/
       defect/*.jpg
       longberry/*.jpg
       peaberry/*.jpg
       premium/*.jpg
     test/*.jpg      # unlabeled -- becomes real_world/ after preprocessing
   ```
3. Run the two DVC-defined stages yourself (see [`dvc.yaml`](../dvc.yaml)):
   ```bash
   python scripts/generate_manifest.py
   python scripts/preprocess_dataset.py
   ```
   This regenerates `metadata/manifest.csv`, `dataset_preprocessed/`, and
   `metadata/manifest_preprocessed.csv` locally -- no R2 access needed for this part.
4. From here, follow [`docs/reproducing-training.md`](reproducing-training.md) to run the
   training/HPO/validation/XAI notebooks against your local `dataset_preprocessed/`.

## 4. Dataset characteristics (from EDA)

Summarized from [`docs/modeling-strategy.md` §1](modeling-strategy.md#1-karakteristik-masalah-dari-eda-bukan-asumsi)
and the EDA reports ([v2](../reports/CBQD%20-%20EDA%20v2%20Report.html),
[v3](../reports/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition)%20Report.html)):

- **Size**: ~1,150-1,211 training images (after cleanup), 4 classes, class ratio 1.033 (near-balanced)
- **Images**: 256x256, one bean per image, plain background, bean occupies ~18-23% of frame
- **Label noise**: ~4.5% of training images are proxy-flagged mislabel candidates (not
  confirmed ground truth) -- see [`docs/preprocessing-recommendations.md` §4](preprocessing-recommendations.md)
- **Label structure**: `defect` is likely not an independent class but a cross-cutting
  "damaged" condition of the other 3 bean types -- see EDA v3 and
  [`docs/modeling-strategy.md` §2](modeling-strategy.md#2-keputusan-framing-flat-4-class-vs-hierarchical)
- **Duplicates**: 11 exact-duplicate groups and multiple near-duplicate pairs (same-class
  and cross-class) required a cluster-aware train/test split to avoid leakage -- see
  [`docs/preprocessing-recommendations.md` §2-3](preprocessing-recommendations.md)

## 5. Provenance / traceability

Every row in `metadata/manifest_preprocessed.csv` carries an `orig_path` back to the raw
Kaggle image it was derived from, and a `cluster_id` from the near-duplicate union-find
(kept even after fold assignment, in case the CV scheme changes later). This is what makes
it possible to audit any prediction, XAI finding, or flagged mislabel candidate back to a
specific original file.
