"""Tests for counting evaluation metrics."""

import json

import pytest

from evaluation.counting_metrics import (
    compute_counting_metrics,
    evaluate_all_videos,
    load_count_json,
)


class TestLoadCountJson:
    def test_loads_inference_format(self, tmp_path):
        data = {
            "video": "test.mp4",
            "total_products": 5,
            "products": [
                {"class_id": 0, "class_name": "Advil", "count": 3},
                {"class_id": 5, "class_name": "Aleve", "count": 2},
            ],
        }
        path = tmp_path / "results.json"
        path.write_text(json.dumps(data))

        counts = load_count_json(path)
        assert counts == {0: 3, 5: 2}

    def test_empty_products(self, tmp_path):
        data = {"products": []}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(data))

        counts = load_count_json(path)
        assert counts == {}


class TestComputeCountingMetrics:
    def test_perfect_match(self):
        preds = {0: 3, 1: 2, 2: 1}
        gt = {0: 3, 1: 2, 2: 1}
        result = compute_counting_metrics(preds, gt)
        assert result["mae"] == 0.0
        assert result["class_accuracy"] == 1.0
        assert result["missed_classes"] == []
        assert result["spurious_classes"] == []
        assert all(e["abs_error"] == 0 for e in result["per_class"])

    def test_overcount(self):
        preds = {0: 5}
        gt = {0: 3}
        result = compute_counting_metrics(preds, gt)
        assert result["per_class"][0]["error"] == 2
        assert result["per_class"][0]["abs_error"] == 2
        assert result["mae"] == 2.0
        assert result["class_accuracy"] == 0.0

    def test_undercount(self):
        preds = {0: 1}
        gt = {0: 4}
        result = compute_counting_metrics(preds, gt)
        assert result["per_class"][0]["error"] == -3
        assert result["per_class"][0]["abs_error"] == 3
        assert result["mae"] == 3.0

    def test_missed_class(self):
        preds = {}
        gt = {0: 2, 1: 3}
        result = compute_counting_metrics(preds, gt)
        assert result["missed_classes"] == [0, 1]
        assert result["spurious_classes"] == []
        # Both classes have abs_error > 0
        assert result["mae"] == (2 + 3) / 2

    def test_spurious_class(self):
        preds = {5: 2, 10: 1}
        gt = {}
        result = compute_counting_metrics(preds, gt)
        assert result["missed_classes"] == []
        assert result["spurious_classes"] == [5, 10]
        assert result["mae"] == (2 + 1) / 2

    def test_mixed_errors(self):
        preds = {0: 3, 1: 0, 2: 2}
        gt = {0: 3, 1: 2}
        result = compute_counting_metrics(preds, gt)
        # class 0: exact match, class 1: undercount by 2, class 2: spurious +2
        assert result["class_accuracy"] == pytest.approx(1 / 3)
        assert result["missed_classes"] == []
        assert result["spurious_classes"] == [2]
        assert result["total_abs_error"] == 4  # 0 + 2 + 2

    def test_both_empty(self):
        result = compute_counting_metrics({}, {})
        assert result["mae"] == 0.0
        assert result["class_accuracy"] == 0.0
        assert result["per_class"] == []


class TestEvaluateAllVideos:
    def _write_results_json(self, path, products):
        data = {"products": products}
        path.write_text(json.dumps(data))

    def test_matches_by_stem(self, tmp_path):
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        self._write_results_json(
            pred_dir / "video1_results.json",
            [{"class_id": 0, "count": 3}],
        )
        self._write_results_json(
            gt_dir / "video1_gt.json",
            [{"class_id": 0, "count": 3}],
        )

        result = evaluate_all_videos(pred_dir, gt_dir)
        assert len(result["per_video"]) == 1
        assert result["per_video"][0]["video"] == "video1"
        assert result["aggregate"]["mae"] == 0.0

    def test_multiple_videos(self, tmp_path):
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        # Video 1: perfect match
        self._write_results_json(
            pred_dir / "v1_results.json",
            [{"class_id": 0, "count": 2}],
        )
        self._write_results_json(
            gt_dir / "v1_gt.json",
            [{"class_id": 0, "count": 2}],
        )

        # Video 2: overcount by 1
        self._write_results_json(
            pred_dir / "v2_results.json",
            [{"class_id": 0, "count": 3}],
        )
        self._write_results_json(
            gt_dir / "v2_gt.json",
            [{"class_id": 0, "count": 2}],
        )

        result = evaluate_all_videos(pred_dir, gt_dir)
        assert result["aggregate"]["total_videos"] == 2
        # Aggregate MAE: (0 + 1) / 2 classes total = 0.5
        assert result["aggregate"]["mae"] == pytest.approx(0.5)

    def test_unmatched_files(self, tmp_path):
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        self._write_results_json(
            pred_dir / "only_pred_results.json",
            [{"class_id": 0, "count": 1}],
        )
        self._write_results_json(
            gt_dir / "only_gt_gt.json",
            [{"class_id": 0, "count": 1}],
        )

        result = evaluate_all_videos(pred_dir, gt_dir)
        assert result["aggregate"]["total_videos"] == 0
        assert "only_pred" in result["aggregate"]["unmatched_predictions"]
        assert "only_gt" in result["aggregate"]["unmatched_ground_truth"]

    def test_empty_directories(self, tmp_path):
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        result = evaluate_all_videos(pred_dir, gt_dir)
        assert result["aggregate"]["total_videos"] == 0
        assert result["aggregate"]["mae"] == 0.0
