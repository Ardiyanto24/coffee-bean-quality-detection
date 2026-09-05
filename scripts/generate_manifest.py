import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_CSV = ROOT_DIR / "metadata" / "manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_train_rows():
    rows = []
    for class_dir in sorted((DATASET_DIR / "train").iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "image_path": image_path.relative_to(DATASET_DIR).as_posix(),
                    "label": class_dir.name,
                    "split": "train",
                })
    return rows


def collect_test_rows():
    rows = []
    for image_path in sorted((DATASET_DIR / "test").iterdir()):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({
                "image_path": image_path.relative_to(DATASET_DIR).as_posix(),
                "label": "",
                "split": "test",
            })
    return rows


def main():
    rows = collect_train_rows() + collect_test_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
