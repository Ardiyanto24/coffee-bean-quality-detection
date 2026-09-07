"""
Preprocessing pipeline implementing docs/preprocessing-recommendations.md.

Reads the raw DVC-tracked dataset (dataset/train/<class>/*.jpg, dataset/test/*.jpg)
and metadata/manifest.csv, then produces:

  dataset_preprocessed/
    train/<class>/*.jpg        -- cluster-aware CV pool (cv_fold 0-3), cropped+resized
    test/<class>/*.jpg         -- held-out LABELED test split (former fold 4), cropped+resized
    real_world/*.jpg           -- the original unlabeled "test" folder, renamed: this is
                                   real-world data with NO known ground truth, NOT an
                                   evaluation set. Carried through the same crop+resize
                                   for consistency, never used to compute accuracy.
  metadata/manifest_preprocessed.csv

Steps (see docs/preprocessing-recommendations.md for rationale of each):
  1. Load manifest.csv, split into labeled train rows and unlabeled real-world rows.
  2. Exact-duplicate (MD5) exclusion -- keep one representative per group.
  3. Near-duplicate (pHash, Hamming<=4) cross-class exclusion within the labeled pool.
  4. Union-Find clustering of same-class near-duplicates that remain, so a single
     physical bean never ends up split across train and the held-out test set.
  5. StratifiedGroupKFold(n_splits=5) on the cleaned labeled pool: fold 4 becomes the
     held-out labeled test set, folds 0-3 remain as the training pool (usable for a
     4-fold CV during model development).
  6. Crop-to-bbox+margin (foreground segmentation) + resize to a standard resolution
     for every retained image, in all three output groups.
  7. Write manifest_preprocessed.csv with full traceability back to the original path.

This script does NOT bake in normalization (mean/std) or any random augmentation --
those stay runtime transforms applied by the training data loader, per
docs/preprocessing-recommendations.md.
"""
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
MANIFEST_PATH = ROOT_DIR / "metadata" / "manifest.csv"
OUT_DIR = ROOT_DIR / "dataset_preprocessed"
OUT_MANIFEST_PATH = ROOT_DIR / "metadata" / "manifest_preprocessed.csv"

NEAR_DUP_THRESH = 4
CROP_MARGIN_FRAC = 0.2
OUT_SIZE = 224
N_FOLDS = 5
TEST_FOLD = 4
SEED = 42


def md5_file(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    img = Image.open(path).convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    d = dct(dct(pixels, axis=0), axis=1)
    low = d[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def crop_to_bbox_with_margin(path, margin_frac=CROP_MARGIN_FRAC, out_size=OUT_SIZE):
    """Foreground threshold segmentation (same heuristic as the EDA notebooks) ->
    crop to bounding box + margin -> resize. Falls back to a plain resize if
    segmentation finds no foreground at all (should not happen on this dataset,
    but keeps the pipeline from crashing on an unexpected input)."""
    img = Image.open(path).convert("RGB")
    gray = np.array(img.convert("L")).astype(np.float32)
    thresh = gray.mean() - 0.6 * gray.std()
    mask = gray < thresh
    if mask.sum() == 0:
        return img.resize((out_size, out_size), Image.LANCZOS)

    h, w = gray.shape
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    margin_y = int((y1 - y0) * margin_frac)
    margin_x = int((x1 - x0) * margin_frac)
    y0 = max(0, y0 - margin_y)
    y1 = min(h, y1 + margin_y)
    x0 = max(0, x0 - margin_x)
    x1 = min(w, x1 + margin_x)
    cropped = img.crop((int(x0), int(y0), int(x1), int(y1)))
    return cropped.resize((out_size, out_size), Image.LANCZOS)


def find_near_dup_pairs(df, thresh=NEAR_DUP_THRESH):
    phashes = np.stack([compute_phash(p) for p in df["abs_path"]])
    packed = np.packbits(phashes, axis=1)
    prefix_keys = packed[:, 0].astype(np.uint16) * 256 + packed[:, 1].astype(np.uint16)
    buckets = {}
    for idx, key in enumerate(prefix_keys):
        buckets.setdefault(int(key), []).append(idx)

    pairs = []
    for idxs in buckets.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                dist = hamming(phashes[a], phashes[b])
                if dist <= thresh:
                    pairs.append((a, b, dist))
    return pd.DataFrame(pairs, columns=["idx_1", "idx_2", "hamming_dist"])


def main():
    manifest_df = pd.read_csv(MANIFEST_PATH)
    manifest_df["abs_path"] = manifest_df["image_path"].apply(lambda p: str(DATASET_DIR / p))

    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    real_world_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    print(f"Loaded manifest: {len(train_df)} labeled (train), {len(real_world_df)} unlabeled (real_world)")

    # ---- Step 2+3: exact-dup and cross-class near-dup exclusion ----
    train_df["md5"] = train_df["abs_path"].apply(md5_file)
    md5_groups = train_df.groupby("md5")["abs_path"].apply(list)
    exact_dup_redundant = set()
    for paths in md5_groups[md5_groups.apply(len) > 1]:
        exact_dup_redundant.update(paths[1:])  # keep first, drop the rest

    pairs_df = find_near_dup_pairs(train_df)
    pairs_df["label_1"] = pairs_df["idx_1"].map(train_df["label"])
    pairs_df["label_2"] = pairs_df["idx_2"].map(train_df["label"])
    pairs_df["path_1"] = pairs_df["idx_1"].map(train_df["abs_path"])
    pairs_df["path_2"] = pairs_df["idx_2"].map(train_df["abs_path"])
    cross_class_pairs = pairs_df[pairs_df["label_1"] != pairs_df["label_2"]]
    cross_class_files = set(cross_class_pairs["path_1"]) | set(cross_class_pairs["path_2"])

    exclude = exact_dup_redundant | cross_class_files
    clean_df = train_df[~train_df["abs_path"].isin(exclude)].reset_index(drop=True)
    print(f"Excluded {len(exclude)} files (exact-dup redundant + cross-class near-dup); "
          f"{len(clean_df)} remain in the labeled pool")

    # ---- Step 4: cluster same-class near-duplicates that remain ----
    same_class_pairs = pairs_df[
        (pairs_df["label_1"] == pairs_df["label_2"])
        & (~pairs_df["path_1"].isin(exclude))
        & (~pairs_df["path_2"].isin(exclude))
    ]
    path_to_clean_idx = {p: i for i, p in enumerate(clean_df["abs_path"])}
    uf = UnionFind(len(clean_df))
    for _, r in same_class_pairs.iterrows():
        if r["path_1"] in path_to_clean_idx and r["path_2"] in path_to_clean_idx:
            uf.union(path_to_clean_idx[r["path_1"]], path_to_clean_idx[r["path_2"]])
    root_to_cluster = {}
    cluster_ids = []
    for i in range(len(clean_df)):
        root = uf.find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        cluster_ids.append(root_to_cluster[root])
    clean_df["cluster_id"] = cluster_ids
    print(f"{clean_df['cluster_id'].nunique()} unique clusters from {len(clean_df)} images")

    # ---- Step 5: cluster-aware stratified split -- fold TEST_FOLD held out as labeled test ----
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    clean_df["cv_fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(clean_df, clean_df["label"], groups=clean_df["cluster_id"])):
        clean_df.loc[clean_df.index[val_idx], "cv_fold"] = fold

    cluster_fold_counts = clean_df.groupby("cluster_id")["cv_fold"].nunique()
    assert (cluster_fold_counts <= 1).all(), "A cluster was split across folds -- near-duplicate leakage risk!"

    new_test_df = clean_df[clean_df["cv_fold"] == TEST_FOLD].copy()
    new_train_df = clean_df[clean_df["cv_fold"] != TEST_FOLD].copy()
    print(f"Held-out LABELED test: {len(new_test_df)} images (former fold {TEST_FOLD})")
    print(f"New training pool: {len(new_train_df)} images (cv_fold 0-{N_FOLDS - 1} minus {TEST_FOLD}, "
          f"usable for {N_FOLDS - 1}-fold CV)")
    print()
    print("Class balance check (train pool):")
    print(new_train_df["label"].value_counts())
    print("Class balance check (held-out test):")
    print(new_test_df["label"].value_counts())

    # ---- Step 6+7: crop+resize every retained image, write to dataset_preprocessed/, build manifest ----
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    records = []

    def process_group(df, split_name, labeled):
        for _, row in df.iterrows():
            label = row["label"] if labeled else ""
            rel_dir = Path(split_name) / label if labeled else Path(split_name)
            out_dir = OUT_DIR / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = Path(row["abs_path"]).name
            out_path = out_dir / out_name
            processed = crop_to_bbox_with_margin(row["abs_path"])
            processed.save(out_path, format="JPEG", quality=92)
            records.append({
                "image_path": str((rel_dir / out_name).as_posix()),
                "label": label,
                "split": split_name,
                "cv_fold": int(row["cv_fold"]) if labeled else None,
                "cluster_id": int(row["cluster_id"]) if labeled else None,
                "orig_path": row["image_path"],
            })

    process_group(new_train_df, "train", labeled=True)
    process_group(new_test_df, "test", labeled=True)
    process_group(real_world_df.assign(abs_path=real_world_df["abs_path"]), "real_world", labeled=False)

    out_manifest = pd.DataFrame(records)
    OUT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.to_csv(OUT_MANIFEST_PATH, index=False)
    print()
    print(f"Wrote {len(out_manifest)} processed images to {OUT_DIR}")
    print(f"Wrote manifest: {OUT_MANIFEST_PATH}")
    print(out_manifest["split"].value_counts())


if __name__ == "__main__":
    main()
