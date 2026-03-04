"""Split product crops into train/val for YOLO classification format.

YOLO classification expects:
    datasets/classification/
    ├── train/
    │   ├── 0_Advil_Liquid_Gel/
    │   └── ...
    └── val/
        ├── 0_Advil_Liquid_Gel/
        └── ...
"""

import random
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig

from data_prep.parse_labels import (
    get_class_mapping,
    get_original_to_yolo_mapping,
    parse_label_file,
)


def split_crops_for_classification(
    crops_dir: Path,
    class_names: dict[int, str],
    original_to_yolo: dict[int, int],
    *,
    output_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[int, int]:
    """Split product crops into train/val directories for YOLO classification.

    Args:
        crops_dir: Directory with subfolders named by 1-indexed original class ID.
        class_names: YOLO 0-indexed class ID to class name mapping.
        original_to_yolo: Original 1-indexed class ID to YOLO 0-indexed mapping.
        output_dir: Root output directory (will contain train/ and val/).
        val_ratio: Fraction of crops for validation.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (num_train, num_val).
    """
    crops_dir = Path(crops_dir)
    output_dir = Path(output_dir)

    num_train = 0
    num_val = 0

    for class_subdir in sorted(crops_dir.iterdir()):
        if not class_subdir.is_dir():
            continue

        # Parse original class ID from folder name (e.g. "00001" → 1)
        try:
            orig_id = int(class_subdir.name)
        except ValueError:
            continue

        if orig_id not in original_to_yolo:
            continue

        yolo_id = original_to_yolo[orig_id]
        folder_name = f"{yolo_id}_{class_names[yolo_id]}"

        # List and shuffle crops
        crops = sorted(class_subdir.glob("*.png"))
        rng = random.Random(seed)
        rng.shuffle(crops)

        n_val = max(1, round(len(crops) * val_ratio))
        val_crops = crops[:n_val]
        train_crops = crops[n_val:]

        # Copy into output directories
        for split, files in [("train", train_crops), ("val", val_crops)]:
            dest = output_dir / split / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dest / src.name)

        num_train += len(train_crops)
        num_val += len(val_crops)

    return num_train, num_val


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(cfg.project_root)
    label_path = project_root / cfg.data.label_file
    crops_dir = project_root / cfg.data.crops_dir
    output_dir = project_root / cfg.data.classification_dir

    labels = parse_label_file(label_path)
    class_names = get_class_mapping(labels)
    original_to_yolo = get_original_to_yolo_mapping(labels)

    print(f"Crops:  {crops_dir} ({len(class_names)} classes)")
    print(f"Output: {output_dir}")

    n_train, n_val = split_crops_for_classification(
        crops_dir,
        class_names,
        original_to_yolo,
        output_dir=output_dir,
    )
    print(f"Done: {n_train} train, {n_val} val ({n_train + n_val} total)")


if __name__ == "__main__":
    main()
