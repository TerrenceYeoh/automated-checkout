"""Tests for synthetic scene composition and YOLO annotation generation."""

import numpy as np
import pytest
from PIL import Image

from data_prep.scene_compositor import (
    compose_scene,
    compute_yolo_annotation,
    load_crop_index,
    paste_product,
    validate_yolo_label,
)


@pytest.fixture
def crops_dir(tmp_path):
    """Create a mini crop directory with 3 classes, 2 images each."""
    for class_id in [0, 1, 2]:
        class_dir = tmp_path / f"{class_id + 1:05d}"
        class_dir.mkdir()
        for i in range(2):
            # Create RGBA crop of varying sizes
            w, h = np.random.randint(30, 80), np.random.randint(30, 80)
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            arr[:, :, :3] = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
            arr[:, :, 3] = 255  # fully opaque
            Image.fromarray(arr, mode="RGBA").save(
                class_dir / f"{class_id + 1:05d}_{i}.png"
            )
    return tmp_path


@pytest.fixture
def background(tmp_path):
    """Create a 640x640 background image."""
    bg = np.full((640, 640, 3), 180, dtype=np.uint8)  # gray conveyor
    path = tmp_path / "bg.jpg"
    Image.fromarray(bg).save(path)
    return path


class TestLoadCropIndex:
    def test_returns_dict(self, crops_dir):
        index = load_crop_index(crops_dir)
        assert isinstance(index, dict)

    def test_correct_class_count(self, crops_dir):
        index = load_crop_index(crops_dir)
        assert len(index) == 3

    def test_correct_image_count(self, crops_dir):
        index = load_crop_index(crops_dir)
        for _class_id, paths in index.items():
            assert len(paths) == 2

    def test_keys_are_class_ids(self, crops_dir):
        index = load_crop_index(crops_dir)
        assert set(index.keys()) == {1, 2, 3}

    def test_ignores_non_numeric_dirs(self, crops_dir):
        """B5: Non-numeric directories like .DS_Store should be skipped."""
        (crops_dir / ".DS_Store").mkdir()
        (crops_dir / "__pycache__").mkdir()
        (crops_dir / "readme.txt").write_text("ignore me")
        index = load_crop_index(crops_dir)
        # Should still only have the 3 numeric class dirs
        assert set(index.keys()) == {1, 2, 3}


class TestPasteProduct:
    def test_returns_modified_scene(self):
        scene = np.full((640, 640, 3), 180, dtype=np.uint8)
        crop = np.zeros((50, 40, 4), dtype=np.uint8)
        crop[:, :, :3] = [255, 0, 0]
        crop[:, :, 3] = 255

        result, bbox = paste_product(scene, crop, x=100, y=200)
        assert result.shape == (640, 640, 3)
        # The pasted area should be red
        assert result[200, 100, 0] == 255

    def test_returns_bbox(self):
        scene = np.full((640, 640, 3), 180, dtype=np.uint8)
        crop = np.zeros((50, 40, 4), dtype=np.uint8)
        crop[:, :, 3] = 255

        _, bbox = paste_product(scene, crop, x=100, y=200)
        # bbox = (x_min, y_min, x_max, y_max)
        assert bbox == (100, 200, 140, 250)

    def test_clips_to_image_bounds(self):
        scene = np.full((640, 640, 3), 180, dtype=np.uint8)
        crop = np.zeros((50, 40, 4), dtype=np.uint8)
        crop[:, :, 3] = 255

        # Place near bottom-right edge — should clip
        _, bbox = paste_product(scene, crop, x=620, y=610)
        x_min, y_min, x_max, y_max = bbox
        assert x_max <= 640
        assert y_max <= 640

    def test_alpha_blending(self):
        scene = np.full((640, 640, 3), 100, dtype=np.uint8)
        crop = np.zeros((50, 40, 4), dtype=np.uint8)
        crop[:, :, :3] = 200
        crop[:25, :, 3] = 255  # top half opaque
        crop[25:, :, 3] = 0  # bottom half transparent

        result, _ = paste_product(scene, crop, x=100, y=200)
        # Top half should be product color
        assert result[200, 100, 0] == 200
        # Bottom half should be background
        assert result[230, 100, 0] == 100


class TestComputeYoloAnnotation:
    def test_normalized_format(self):
        bbox = (100, 200, 300, 400)
        img_w, img_h = 640, 640
        class_id = 5
        ann = compute_yolo_annotation(bbox, class_id, img_w, img_h)
        # Format: "class_id x_center y_center width height" (normalized)
        parts = ann.split()
        assert len(parts) == 5
        assert int(parts[0]) == 5

    def test_center_calculation(self):
        bbox = (100, 200, 300, 400)
        ann = compute_yolo_annotation(bbox, 0, 640, 640)
        parts = ann.split()
        x_center = float(parts[1])
        y_center = float(parts[2])
        assert abs(x_center - 200 / 640) < 1e-6
        assert abs(y_center - 300 / 640) < 1e-6

    def test_width_height_calculation(self):
        bbox = (100, 200, 300, 400)
        ann = compute_yolo_annotation(bbox, 0, 640, 640)
        parts = ann.split()
        width = float(parts[3])
        height = float(parts[4])
        assert abs(width - 200 / 640) < 1e-6
        assert abs(height - 200 / 640) < 1e-6

    def test_all_values_in_0_1_range(self):
        bbox = (0, 0, 640, 640)
        ann = compute_yolo_annotation(bbox, 0, 640, 640)
        parts = ann.split()
        for val in parts[1:]:
            assert 0.0 <= float(val) <= 1.0


class TestValidateYoloLabel:
    def test_valid_label(self):
        assert validate_yolo_label("0 0.5 0.5 0.2 0.3") is True

    def test_invalid_class(self):
        assert validate_yolo_label("-1 0.5 0.5 0.2 0.3") is False

    def test_out_of_range_coords(self):
        assert validate_yolo_label("0 1.5 0.5 0.2 0.3") is False

    def test_wrong_field_count(self):
        assert validate_yolo_label("0 0.5 0.5") is False

    def test_zero_width(self):
        assert validate_yolo_label("0 0.5 0.5 0.0 0.3") is False


class TestComposeScene:
    def test_returns_image_and_annotations(self, crops_dir, background):
        bg = np.array(Image.open(background))
        index = load_crop_index(crops_dir)
        scene, annotations = compose_scene(
            bg, index, num_products=3, class_id_to_yolo={1: 0, 2: 1, 3: 2}
        )
        assert scene.shape == (640, 640, 3)
        assert len(annotations) > 0

    def test_annotation_format(self, crops_dir, background):
        bg = np.array(Image.open(background))
        index = load_crop_index(crops_dir)
        _, annotations = compose_scene(
            bg, index, num_products=2, class_id_to_yolo={1: 0, 2: 1, 3: 2}
        )
        for ann in annotations:
            assert validate_yolo_label(ann)

    def test_respects_num_products(self, crops_dir, background):
        bg = np.array(Image.open(background))
        index = load_crop_index(crops_dir)
        _, annotations = compose_scene(
            bg, index, num_products=2, class_id_to_yolo={1: 0, 2: 1, 3: 2}
        )
        # Should have exactly 2 annotations (unless clipping removes one)
        assert len(annotations) <= 2

    def test_scene_is_modified(self, crops_dir, background):
        bg = np.array(Image.open(background)).copy()
        original = bg.copy()
        index = load_crop_index(crops_dir)
        scene, _ = compose_scene(
            bg, index, num_products=3, class_id_to_yolo={1: 0, 2: 1, 3: 2}
        )
        # Scene should differ from the original background
        assert not np.array_equal(scene, original)
