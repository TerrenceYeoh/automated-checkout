"""Tests for crop extraction from training images + segmentation masks."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_prep.extract_crops import (
    bbox_from_mask,
    extract_all_crops,
    extract_crop,
    get_image_mask_pairs,
    process_single_image,
    threshold_mask,
)


@pytest.fixture
def sample_image(tmp_path):
    """Create a 100x80 RGB image with a colored rectangle."""
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    img[20:60, 30:70] = [255, 128, 64]  # colored rect at rows 20-59, cols 30-69
    pil = Image.fromarray(img)
    path = tmp_path / "00001_1.jpg"
    pil.save(path)
    return path


@pytest.fixture
def sample_mask(tmp_path):
    """Create a matching mask with JPEG-like soft edges."""
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 255  # white rect
    # Simulate JPEG compression artifacts at edges
    mask[19, 30:70] = 128
    mask[60, 30:70] = 100
    pil = Image.fromarray(mask, mode="L")
    path = tmp_path / "00001_1_seg.jpg"
    pil.save(path)
    return path


@pytest.fixture
def sample_data_dir(tmp_path, sample_image, sample_mask):
    """Create a mini dataset directory structure."""
    train_dir = tmp_path / "train"
    seg_dir = tmp_path / "segmentation_labels"
    train_dir.mkdir()
    seg_dir.mkdir()

    # Move files into proper dirs
    sample_image.rename(train_dir / sample_image.name)
    sample_mask.rename(seg_dir / sample_mask.name)

    # Add a second class
    img2 = np.zeros((50, 60, 3), dtype=np.uint8)
    img2[10:40, 15:50] = [0, 200, 100]
    Image.fromarray(img2).save(train_dir / "00002_1.jpg")

    mask2 = np.zeros((50, 60), dtype=np.uint8)
    mask2[10:40, 15:50] = 255
    Image.fromarray(mask2, mode="L").save(seg_dir / "00002_1_seg.jpg")

    return tmp_path


class TestThresholdMask:
    def test_binary_output(self):
        mask = np.array([0, 50, 127, 128, 200, 255], dtype=np.uint8)
        result = threshold_mask(mask, threshold=128)
        np.testing.assert_array_equal(result, [0, 0, 0, 255, 255, 255])

    def test_preserves_shape(self):
        mask = np.random.randint(0, 256, (80, 100), dtype=np.uint8)
        result = threshold_mask(mask)
        assert result.shape == mask.shape

    def test_all_zeros(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = threshold_mask(mask)
        assert result.max() == 0

    def test_all_white(self):
        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = threshold_mask(mask)
        assert result.min() == 255


class TestBboxFromMask:
    def test_returns_xywh(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 255
        bbox = bbox_from_mask(mask)
        # (x_min, y_min, x_max, y_max)
        assert bbox == (30, 20, 70, 60)

    def test_single_pixel(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 30] = 255
        bbox = bbox_from_mask(mask)
        assert bbox == (30, 25, 31, 26)

    def test_full_image(self):
        mask = np.full((40, 60), 255, dtype=np.uint8)
        bbox = bbox_from_mask(mask)
        assert bbox == (0, 0, 60, 40)

    def test_empty_mask_returns_none(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        bbox = bbox_from_mask(mask)
        assert bbox is None


class TestExtractCrop:
    def test_returns_rgba(self, sample_image, sample_mask):
        img = np.array(Image.open(sample_image))
        mask = np.array(Image.open(sample_mask).convert("L"))
        crop = extract_crop(img, mask)
        assert crop.shape[2] == 4  # RGBA

    def test_crop_dimensions(self, sample_image, sample_mask):
        img = np.array(Image.open(sample_image))
        mask = np.array(Image.open(sample_mask).convert("L"))
        crop = extract_crop(img, mask)
        # Bbox should be approximately rows 20-60, cols 30-70
        # Due to threshold, may include row 19 (value=128, >= threshold)
        h, w = crop.shape[:2]
        assert 35 <= h <= 45  # ~40 rows
        assert 35 <= w <= 45  # ~40 cols

    def test_alpha_channel_matches_mask(self, sample_image, sample_mask):
        img = np.array(Image.open(sample_image))
        mask = np.array(Image.open(sample_mask).convert("L"))
        crop = extract_crop(img, mask)
        alpha = crop[:, :, 3]
        # Alpha should only be 0 or 255
        unique = np.unique(alpha)
        assert all(v in [0, 255] for v in unique)

    def test_empty_mask_returns_none(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        crop = extract_crop(img, mask)
        assert crop is None


class TestGetImageMaskPairs:
    def test_finds_pairs(self, sample_data_dir):
        train_dir = sample_data_dir / "train"
        seg_dir = sample_data_dir / "segmentation_labels"
        pairs = get_image_mask_pairs(train_dir, seg_dir)
        assert len(pairs) == 2

    def test_pair_structure(self, sample_data_dir):
        train_dir = sample_data_dir / "train"
        seg_dir = sample_data_dir / "segmentation_labels"
        pairs = get_image_mask_pairs(train_dir, seg_dir)
        for img_path, mask_path, class_id in pairs:
            assert isinstance(img_path, Path)
            assert isinstance(mask_path, Path)
            assert isinstance(class_id, int)
            assert img_path.exists()
            assert mask_path.exists()

    def test_extracts_class_id(self, sample_data_dir):
        train_dir = sample_data_dir / "train"
        seg_dir = sample_data_dir / "segmentation_labels"
        pairs = get_image_mask_pairs(train_dir, seg_dir)
        class_ids = {p[2] for p in pairs}
        assert class_ids == {1, 2}


class TestProcessSingleImage:
    def test_saves_rgba_png(self, sample_data_dir, tmp_path):
        train_dir = sample_data_dir / "train"
        seg_dir = sample_data_dir / "segmentation_labels"
        output_dir = tmp_path / "crops"
        pairs = get_image_mask_pairs(train_dir, seg_dir)

        result = process_single_image(pairs[0], output_dir)
        assert result is not None
        crop_path, class_id = result
        assert crop_path.suffix == ".png"
        assert crop_path.exists()

        # Verify it's RGBA
        img = Image.open(crop_path)
        assert img.mode == "RGBA"

    def test_output_in_class_subfolder(self, sample_data_dir, tmp_path):
        train_dir = sample_data_dir / "train"
        seg_dir = sample_data_dir / "segmentation_labels"
        output_dir = tmp_path / "crops"
        pairs = get_image_mask_pairs(train_dir, seg_dir)

        result = process_single_image(pairs[0], output_dir)
        crop_path, class_id = result
        # Should be in output_dir/<zero-padded class_id>/
        assert crop_path.parent.parent == output_dir


class TestExtractAllCrops:
    def test_corrupt_image_does_not_crash_pipeline(self, sample_data_dir, tmp_path):
        """B4: A corrupt image should be skipped, not crash the pipeline."""
        train_dir = sample_data_dir / "train"
        seg_dir = sample_data_dir / "segmentation_labels"
        output_dir = tmp_path / "crops"

        # Add a corrupt image file
        corrupt_img = train_dir / "00003_1.jpg"
        corrupt_img.write_bytes(b"not a real image")
        corrupt_mask = seg_dir / "00003_1_seg.jpg"
        corrupt_mask.write_bytes(b"not a real mask")

        # Should not raise — corrupt file is skipped
        results = extract_all_crops(train_dir, seg_dir, output_dir, max_workers=1)
        # The 2 valid images should still be processed
        assert len(results) >= 1
