"""Split composite scene dataset into train/val/test sets for YOLO training."""

import random
import shutil
from collections import Counter
from pathlib import Path


def compute_class_distribution(labels_dir: Path) -> dict[int, int]:
    """Count instances of each class in a labels directory.

    Args:
        labels_dir: Directory containing YOLO-format label files.

    Returns:
        Dict mapping class_id to instance count.
    """
    counts: dict[int, int] = Counter()
    for label_path in Path(labels_dir).glob("*.txt"):
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    counts[int(parts[0])] += 1
                except (ValueError, IndexError):
                    continue
    return dict(sorted(counts.items()))


def _print_distribution_summary(train_dir: Path, val_dir: Path, test_dir: Path) -> None:
    """Print class distribution summary and imbalance mitigation proposals."""
    for name, directory in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        dist = compute_class_distribution(directory)
        if not dist:
            continue
        values = list(dist.values())
        mean_count = sum(values) / len(values)
        min_count = min(values)
        max_count = max(values)
        std_dev = (sum((v - mean_count) ** 2 for v in values) / len(values)) ** 0.5
        ratio = max_count / min_count if min_count > 0 else float("inf")
        print(
            f"  {name}: {len(dist)} classes, "
            f"min={min_count}, max={max_count}, mean={mean_count:.1f}, "
            f"std={std_dev:.1f}, max/min ratio={ratio:.2f}"
        )

    print("\nClass imbalance mitigation (available if needed):")
    print("  1. Per-class loss weighting — tune global cls loss weight")
    print(
        "  2. Oversampling during scene generation — weight rare classes in compositor"
    )
    print("  3. Focal loss — already enabled in YOLO classification head")
    print("  4. Copy-paste augmentation — already enabled at copy_paste=0.1")


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int | None = None,
) -> tuple[int, int, int]:
    """Split images and labels into stratified train/val/test sets.

    Stratification assigns each image a key based on its least globally-frequent
    class, then splits within each group to ensure all classes are proportionally
    represented across splits.

    Files are copied (not moved) to preserve the originals.

    Args:
        images_dir: Directory containing scene images.
        labels_dir: Directory containing YOLO label files.
        output_dir: Root output directory. Creates images/{train,val,test}
            and labels/{train,val,test}.
        val_ratio: Fraction of data to use for validation.
        test_ratio: Fraction of data to use for testing.
        seed: Random seed for reproducibility.

    Returns:
        (num_train, num_val, num_test) tuple.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Collect all image stems that have matching labels, and parse their classes
    image_files = sorted(images_dir.glob("*.jpg"))
    stem_classes: dict[str, set[int]] = {}
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            classes = set()
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        classes.add(int(parts[0]))
                    except (ValueError, IndexError):
                        continue
            stem_classes[img_path.stem] = classes

    # Count global class frequencies
    global_freq: dict[int, int] = Counter()
    for classes in stem_classes.values():
        for cls_id in classes:
            global_freq[cls_id] += 1

    # Group images by stratification key (least frequent class in each image)
    groups: dict[int, list[str]] = {}
    for stem, classes in stem_classes.items():
        if classes:
            strat_key = min(classes, key=lambda c: global_freq[c])
        else:
            strat_key = -1  # images with no classes
        groups.setdefault(strat_key, []).append(stem)

    # Shuffle and split within each group
    rng = random.Random(seed) if seed is not None else random.Random()

    train_stems: list[str] = []
    val_stems: list[str] = []
    test_stems: list[str] = []

    for key in sorted(groups.keys()):
        members = groups[key]
        rng.shuffle(members)
        n = len(members)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test

        # Safety: if group is too small, ensure at least 1 per split
        if n_train < 1 and n >= 3:
            n_train = 1
            n_val = max(1, n - n_train - n_test)
            n_test = n - n_train - n_val

        test_stems.extend(members[:n_test])
        val_stems.extend(members[n_test : n_test + n_val])
        train_stems.extend(members[n_test + n_val :])

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for stem in train_stems:
        shutil.copy2(
            images_dir / f"{stem}.jpg", output_dir / "images" / "train" / f"{stem}.jpg"
        )
        shutil.copy2(
            labels_dir / f"{stem}.txt", output_dir / "labels" / "train" / f"{stem}.txt"
        )

    for stem in val_stems:
        shutil.copy2(
            images_dir / f"{stem}.jpg", output_dir / "images" / "val" / f"{stem}.jpg"
        )
        shutil.copy2(
            labels_dir / f"{stem}.txt", output_dir / "labels" / "val" / f"{stem}.txt"
        )

    for stem in test_stems:
        shutil.copy2(
            images_dir / f"{stem}.jpg", output_dir / "images" / "test" / f"{stem}.jpg"
        )
        shutil.copy2(
            labels_dir / f"{stem}.txt", output_dir / "labels" / "test" / f"{stem}.txt"
        )

    print(
        f"Split complete: {len(train_stems)} train, "
        f"{len(val_stems)} val, {len(test_stems)} test"
    )
    print("\nClass distribution per split:")
    _print_distribution_summary(
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        output_dir / "labels" / "test",
    )

    return len(train_stems), len(val_stems), len(test_stems)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root / "datasets" / "checkout"

    split_dataset(
        images_dir=dataset_dir / "images" / "all",
        labels_dir=dataset_dir / "labels" / "all",
        output_dir=dataset_dir,
        val_ratio=0.15,
        test_ratio=0.10,
        seed=42,
    )
