"""Tests for visualization utilities."""

import numpy as np
import pytest

from utils.visualization import draw_yolo_boxes


@pytest.fixture
def sample_image():
    """480x640 black BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def label_file(tmp_path):
    """YOLO label file with one centered box."""
    f = tmp_path / "labels.txt"
    # class 0, centered at (0.5, 0.5), width 0.2, height 0.3
    f.write_text("0 0.5 0.5 0.2 0.3\n")
    return f


class TestDrawYoloBoxes:
    def test_returns_image_same_shape(self, sample_image, label_file):
        result = draw_yolo_boxes(sample_image, label_file)
        assert result.shape == sample_image.shape

    def test_does_not_modify_original(self, sample_image, label_file):
        original = sample_image.copy()
        draw_yolo_boxes(sample_image, label_file)
        np.testing.assert_array_equal(sample_image, original)

    def test_draws_on_image(self, sample_image, label_file):
        result = draw_yolo_boxes(sample_image, label_file)
        # The result should differ from blank black image (boxes were drawn)
        assert not np.array_equal(result, sample_image)

    def test_missing_label_file_returns_copy(self, sample_image, tmp_path):
        result = draw_yolo_boxes(sample_image, tmp_path / "nonexistent.txt")
        np.testing.assert_array_equal(result, sample_image)

    def test_malformed_lines_skipped(self, sample_image, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("bad line\n0 0.5 0.5 0.2 0.3\nnot a number 0.1 0.2 0.3 0.4\n")
        result = draw_yolo_boxes(sample_image, f)
        # Should still draw the one valid box without crashing
        assert not np.array_equal(result, sample_image)

    def test_class_names_displayed(self, sample_image, label_file):
        class_names = {0: "Advil"}
        result = draw_yolo_boxes(sample_image, label_file, class_names=class_names)
        # Should draw without error; visual check not possible but ensures no crash
        assert result.shape == sample_image.shape

    def test_box_coordinates(self, sample_image, label_file):
        """Verify box is drawn in the expected region."""
        result = draw_yolo_boxes(sample_image, label_file, color=(0, 255, 0))
        # Expected box: center (320, 240), size (128, 144)
        # x1=256, y1=168, x2=384, y2=312
        # Check that pixels on the box edge are non-zero (green)
        assert result[168, 320, 1] > 0  # top edge, green channel
