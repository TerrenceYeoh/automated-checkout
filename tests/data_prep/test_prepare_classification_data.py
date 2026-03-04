"""Tests for classification dataset preparation (crop splitting)."""

import pytest

from data_prep.prepare_classification_data import split_crops_for_classification


@pytest.fixture
def crops_dir(tmp_path):
    """Create a mini crop directory with 3 original classes, 10 crops each.

    Mimics data/crops/ layout: subfolders named by 1-indexed original class ID.
    """
    for orig_id in [1, 2, 3]:
        class_dir = tmp_path / "crops" / f"{orig_id:05d}"
        class_dir.mkdir(parents=True)
        for i in range(10):
            (class_dir / f"{orig_id:05d}_{i}.png").write_bytes(b"fake-png")
    return tmp_path / "crops"


@pytest.fixture
def class_names():
    """YOLO 0-indexed class names (from get_class_mapping)."""
    return {0: "Advil_Liquid_Gel", 1: "Advil_Tablets", 2: "Woolite"}


@pytest.fixture
def original_to_yolo():
    """Mapping from original 1-indexed IDs to YOLO 0-indexed IDs."""
    return {1: 0, 2: 1, 3: 2}


class TestSplitCropsForClassification:
    def test_creates_train_val_dirs(
        self, crops_dir, class_names, original_to_yolo, tmp_path
    ):
        output_dir = tmp_path / "classification"
        split_crops_for_classification(
            crops_dir,
            class_names,
            original_to_yolo,
            output_dir=output_dir,
        )
        assert (output_dir / "train").is_dir()
        assert (output_dir / "val").is_dir()

    def test_correct_split_ratio(
        self, crops_dir, class_names, original_to_yolo, tmp_path
    ):
        output_dir = tmp_path / "classification"
        n_train, n_val = split_crops_for_classification(
            crops_dir,
            class_names,
            original_to_yolo,
            output_dir=output_dir,
            val_ratio=0.15,
        )
        # 10 crops per class, 15% val → 1 or 2 val per class, 8 or 9 train
        total = n_train + n_val
        assert total == 30  # 3 classes × 10
        assert n_val == pytest.approx(30 * 0.15, abs=3)

    def test_all_crops_accounted_for(
        self, crops_dir, class_names, original_to_yolo, tmp_path
    ):
        output_dir = tmp_path / "classification"
        n_train, n_val = split_crops_for_classification(
            crops_dir,
            class_names,
            original_to_yolo,
            output_dir=output_dir,
        )
        # Count actual files
        train_files = list((output_dir / "train").rglob("*.png"))
        val_files = list((output_dir / "val").rglob("*.png"))
        assert len(train_files) == n_train
        assert len(val_files) == n_val
        assert len(train_files) + len(val_files) == 30

    def test_no_overlap(self, crops_dir, class_names, original_to_yolo, tmp_path):
        output_dir = tmp_path / "classification"
        split_crops_for_classification(
            crops_dir,
            class_names,
            original_to_yolo,
            output_dir=output_dir,
        )
        train_names = {f.name for f in (output_dir / "train").rglob("*.png")}
        val_names = {f.name for f in (output_dir / "val").rglob("*.png")}
        assert train_names.isdisjoint(val_names)

    def test_class_folders_use_yolo_ids(
        self, crops_dir, class_names, original_to_yolo, tmp_path
    ):
        output_dir = tmp_path / "classification"
        split_crops_for_classification(
            crops_dir,
            class_names,
            original_to_yolo,
            output_dir=output_dir,
        )
        train_dirs = sorted(d.name for d in (output_dir / "train").iterdir())
        expected = ["0_Advil_Liquid_Gel", "1_Advil_Tablets", "2_Woolite"]
        assert train_dirs == expected

    def test_reproducible_with_seed(
        self, crops_dir, class_names, original_to_yolo, tmp_path
    ):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        split_crops_for_classification(
            crops_dir,
            class_names,
            original_to_yolo,
            output_dir=out1,
            seed=42,
        )
        split_crops_for_classification(
            crops_dir,
            class_names,
            original_to_yolo,
            output_dir=out2,
            seed=42,
        )
        val1 = sorted(f.name for f in (out1 / "val").rglob("*.png"))
        val2 = sorted(f.name for f in (out2 / "val").rglob("*.png"))
        assert val1 == val2
