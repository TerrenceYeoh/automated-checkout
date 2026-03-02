"""Tests for label parsing and YOLO data.yaml generation."""

from pathlib import Path

import pytest
import yaml

from data_prep.parse_labels import (
    generate_data_yaml,
    get_class_mapping,
    get_original_to_yolo_mapping,
    parse_label_file,
)

SAMPLE_LABELS = """\
Advil Liquid Gel,1
Advil Tablets,2
Alcohol Pads,3
"""

SAMPLE_LABELS_FULL = """\
Advil Liquid Gel,1
Advil Tablets,2
Alcohol Pads,3
Woolite Delicates _Attempt_,116
"""


@pytest.fixture
def label_file(tmp_path):
    """Create a temporary label file."""
    f = tmp_path / "label.txt"
    f.write_text(SAMPLE_LABELS)
    return f


@pytest.fixture
def label_file_full(tmp_path):
    """Create a label file with non-contiguous IDs to test mapping."""
    f = tmp_path / "label.txt"
    f.write_text(SAMPLE_LABELS_FULL)
    return f


class TestParseLabelFile:
    def test_returns_dict(self, label_file):
        result = parse_label_file(label_file)
        assert isinstance(result, dict)

    def test_correct_count(self, label_file):
        result = parse_label_file(label_file)
        assert len(result) == 3

    def test_maps_id_to_name(self, label_file):
        result = parse_label_file(label_file)
        assert result[1] == "Advil Liquid Gel"
        assert result[2] == "Advil Tablets"
        assert result[3] == "Alcohol Pads"

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "label.txt"
        f.write_text("Advil Liquid Gel,1\n\n\nAlcohol Pads,3\n")
        result = parse_label_file(f)
        assert len(result) == 2
        assert result[1] == "Advil Liquid Gel"
        assert result[3] == "Alcohol Pads"

    def test_handles_special_characters(self, tmp_path):
        f = tmp_path / "label.txt"
        f.write_text("Mac 'n' Cheese Shells,65\nM&Ms Peanuts,73\n")
        result = parse_label_file(f)
        assert result[65] == "Mac 'n' Cheese Shells"
        assert result[73] == "M&Ms Peanuts"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_label_file(Path("/nonexistent/label.txt"))

    def test_skips_malformed_class_ids(self, tmp_path):
        """B8: Non-integer class IDs should be skipped, not crash."""
        f = tmp_path / "label.txt"
        f.write_text("Advil Liquid Gel,1\nBad Product,abc\nAlcohol Pads,3\n")
        result = parse_label_file(f)
        assert len(result) == 2
        assert result[1] == "Advil Liquid Gel"
        assert result[3] == "Alcohol Pads"

    def test_duplicate_class_ids_warns(self, tmp_path):
        """B9: Duplicate class IDs should produce a warning."""
        f = tmp_path / "label.txt"
        f.write_text("Advil Liquid Gel,1\nAdvil Tablets,1\nAlcohol Pads,3\n")
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_label_file(f)
            assert len(w) == 1
            assert "Duplicate" in str(w[0].message)
        # Last value wins
        assert result[1] == "Advil Tablets"
        assert len(result) == 2


class TestGetClassMapping:
    """Test the 1-indexed to 0-indexed YOLO class mapping."""

    def test_zero_indexed(self, label_file):
        labels = parse_label_file(label_file)
        mapping = get_class_mapping(labels)
        # YOLO uses 0-indexed classes
        assert 0 in mapping
        assert 1 in mapping
        assert 2 in mapping

    def test_preserves_order(self, label_file):
        labels = parse_label_file(label_file)
        mapping = get_class_mapping(labels)
        assert mapping[0] == "Advil Liquid Gel"
        assert mapping[1] == "Advil Tablets"
        assert mapping[2] == "Alcohol Pads"

    def test_non_contiguous_ids(self, label_file_full):
        """Even if original IDs are non-contiguous, YOLO mapping is 0..N-1."""
        labels = parse_label_file(label_file_full)
        mapping = get_class_mapping(labels)
        assert len(mapping) == 4
        # Should be sorted by original ID and remapped 0..3
        assert mapping[0] == "Advil Liquid Gel"
        assert mapping[3] == "Woolite Delicates _Attempt_"

    def test_mapping_count_matches_labels(self, label_file_full):
        """YOLO mapping should have same number of entries as original labels."""
        labels = parse_label_file(label_file_full)
        mapping = get_class_mapping(labels)
        assert len(mapping) == len(labels)


class TestGetOriginalToYoloMapping:
    def test_contiguous_ids(self, label_file):
        labels = parse_label_file(label_file)
        mapping = get_original_to_yolo_mapping(labels)
        # Original IDs 1,2,3 -> YOLO 0,1,2
        assert mapping == {1: 0, 2: 1, 3: 2}

    def test_non_contiguous_ids(self, label_file_full):
        labels = parse_label_file(label_file_full)
        mapping = get_original_to_yolo_mapping(labels)
        # Original IDs 1,2,3,116 -> YOLO 0,1,2,3
        assert mapping == {1: 0, 2: 1, 3: 2, 116: 3}

    def test_consistent_with_get_class_mapping(self, label_file_full):
        """Both mapping functions should agree on the ordering."""
        labels = parse_label_file(label_file_full)
        class_mapping = get_class_mapping(labels)
        orig_to_yolo = get_original_to_yolo_mapping(labels)

        for orig_id, yolo_id in orig_to_yolo.items():
            assert class_mapping[yolo_id] == labels[orig_id]


class TestGenerateDataYaml:
    def test_creates_file(self, label_file, tmp_path):
        labels = parse_label_file(label_file)
        mapping = get_class_mapping(labels)
        output = tmp_path / "data.yaml"
        generate_data_yaml(mapping, output, dataset_path="./datasets/checkout")
        assert output.exists()

    def test_valid_yaml(self, label_file, tmp_path):
        labels = parse_label_file(label_file)
        mapping = get_class_mapping(labels)
        output = tmp_path / "data.yaml"
        generate_data_yaml(mapping, output, dataset_path="./datasets/checkout")
        with open(output) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_yaml_structure(self, label_file, tmp_path):
        labels = parse_label_file(label_file)
        mapping = get_class_mapping(labels)
        output = tmp_path / "data.yaml"
        generate_data_yaml(mapping, output, dataset_path="./datasets/checkout")
        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["path"] == "./datasets/checkout"
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
        assert "test" not in data
        assert data["nc"] == 3
        assert data["names"][0] == "Advil Liquid Gel"
        assert data["names"][1] == "Advil Tablets"
        assert data["names"][2] == "Alcohol Pads"

    def test_include_test(self, label_file, tmp_path):
        labels = parse_label_file(label_file)
        mapping = get_class_mapping(labels)
        output = tmp_path / "data.yaml"
        generate_data_yaml(
            mapping, output, dataset_path="./datasets/checkout", include_test=True
        )
        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["test"] == "images/test"
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
