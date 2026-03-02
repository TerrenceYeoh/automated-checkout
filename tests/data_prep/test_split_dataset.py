"""Tests for stratified train/val/test dataset splitting."""

import shutil

import pytest
from PIL import Image

from data_prep.split_dataset import compute_class_distribution, split_dataset

NUM_CLASSES = 5
NUM_IMAGES = 40


@pytest.fixture
def scene_dataset(tmp_path):
    """Create a mock scene dataset with images and multi-class labels."""
    images_dir = tmp_path / "images" / "all"
    labels_dir = tmp_path / "labels" / "all"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(NUM_IMAGES):
        name = f"scene_{i:06d}"
        Image.new("RGB", (640, 640)).save(images_dir / f"{name}.jpg")
        # Assign 1-3 classes per image, cycling through classes
        cls_a = i % NUM_CLASSES
        cls_b = (i + 1) % NUM_CLASSES
        lines = [
            f"{cls_a} 0.5 0.5 0.2 0.3",
            f"{cls_b} 0.3 0.3 0.1 0.2",
        ]
        if i % 3 == 0:
            cls_c = (i + 2) % NUM_CLASSES
            lines.append(f"{cls_c} 0.7 0.7 0.15 0.25")
        (labels_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")

    return tmp_path


class TestSplitDataset:
    def test_creates_train_val_test_dirs(self, scene_dataset):
        split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
        )
        for split in ["train", "val", "test"]:
            assert (scene_dataset / "images" / split).exists()
            assert (scene_dataset / "labels" / split).exists()

    def test_correct_split_ratio(self, scene_dataset):
        n_train, n_val, n_test = split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
        )
        total = n_train + n_val + n_test
        assert total == NUM_IMAGES
        # Verify approximate ratios (stratification may shift counts slightly)
        assert n_train > n_val
        assert n_train > n_test
        assert n_val > 0
        assert n_test > 0

    def test_labels_match_images(self, scene_dataset):
        split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
        )
        for split in ["train", "val", "test"]:
            img_stems = {
                p.stem for p in (scene_dataset / "images" / split).glob("*.jpg")
            }
            lbl_stems = {
                p.stem for p in (scene_dataset / "labels" / split).glob("*.txt")
            }
            assert img_stems == lbl_stems

    def test_no_overlap(self, scene_dataset):
        split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
        )
        train_stems = {
            p.stem for p in (scene_dataset / "images" / "train").glob("*.jpg")
        }
        val_stems = {p.stem for p in (scene_dataset / "images" / "val").glob("*.jpg")}
        test_stems = {p.stem for p in (scene_dataset / "images" / "test").glob("*.jpg")}
        assert train_stems.isdisjoint(val_stems)
        assert train_stems.isdisjoint(test_stems)
        assert val_stems.isdisjoint(test_stems)
        assert len(train_stems) + len(val_stems) + len(test_stems) == NUM_IMAGES

    def test_reproducible_with_seed(self, scene_dataset):
        split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
            seed=42,
        )
        train_1 = {p.stem for p in (scene_dataset / "images" / "train").glob("*.jpg")}
        val_1 = {p.stem for p in (scene_dataset / "images" / "val").glob("*.jpg")}
        test_1 = {p.stem for p in (scene_dataset / "images" / "test").glob("*.jpg")}

        # Clean up and redo
        for split in ["train", "val", "test"]:
            shutil.rmtree(scene_dataset / "images" / split)
            shutil.rmtree(scene_dataset / "labels" / split)

        split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
            seed=42,
        )
        train_2 = {p.stem for p in (scene_dataset / "images" / "train").glob("*.jpg")}
        val_2 = {p.stem for p in (scene_dataset / "images" / "val").glob("*.jpg")}
        test_2 = {p.stem for p in (scene_dataset / "images" / "test").glob("*.jpg")}

        assert train_1 == train_2
        assert val_1 == val_2
        assert test_1 == test_2

    def test_returns_three_counts(self, scene_dataset):
        result = split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
            seed=42,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        n_train, n_val, n_test = result
        assert isinstance(n_train, int)
        assert isinstance(n_val, int)
        assert isinstance(n_test, int)
        assert n_train + n_val + n_test == NUM_IMAGES

    def test_stratification_distributes_classes(self, scene_dataset):
        """All classes present in source should appear in each split."""
        split_dataset(
            images_dir=scene_dataset / "images" / "all",
            labels_dir=scene_dataset / "labels" / "all",
            output_dir=scene_dataset,
            val_ratio=0.15,
            test_ratio=0.10,
            seed=42,
        )
        source_dist = compute_class_distribution(scene_dataset / "labels" / "all")
        source_classes = set(source_dist.keys())

        for split in ["train", "val", "test"]:
            split_dist = compute_class_distribution(scene_dataset / "labels" / split)
            split_classes = set(split_dist.keys())
            assert split_classes == source_classes, (
                f"{split} missing classes: {source_classes - split_classes}"
            )


class TestComputeClassDistribution:
    def test_counts_instances(self, scene_dataset):
        dist = compute_class_distribution(scene_dataset / "labels" / "all")
        assert isinstance(dist, dict)
        assert len(dist) == NUM_CLASSES
        # Every class should have instances
        for cls_id in range(NUM_CLASSES):
            assert dist[cls_id] > 0

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty_labels"
        empty.mkdir()
        dist = compute_class_distribution(empty)
        assert dist == {}

    def test_correct_total(self, scene_dataset):
        """Total instances should match what we put in the labels."""
        dist = compute_class_distribution(scene_dataset / "labels" / "all")
        total = sum(dist.values())
        # Each image has 2 classes, plus every 3rd image has a 3rd class
        # 40 images * 2 = 80, plus 14 images with 3rd class (0,3,6,...,39) = 14
        expected = NUM_IMAGES * 2 + (NUM_IMAGES + 2) // 3
        assert total == expected
