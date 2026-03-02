"""Tests for detection evaluation metrics."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.detection_metrics import format_detection_results, run_detection_eval


@pytest.fixture
def mock_val_results():
    """Create a mock YOLO validation results object."""
    results = MagicMock()
    results.results_dict = {
        "metrics/mAP50(B)": 0.85,
        "metrics/mAP50-95(B)": 0.62,
        "metrics/precision(B)": 0.78,
        "metrics/recall(B)": 0.72,
    }
    results.box.ap50 = np.array([0.95, 0.30, 0.88, 0.10, 0.72])
    results.box.ap_class_index = np.array([0, 1, 2, 3, 4])
    return results


class TestRunDetectionEval:
    @patch("evaluation.detection_metrics.YOLO")
    def test_calls_model_val_with_correct_args(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.val.return_value = MagicMock()

        eval_cfg = {
            "split": "val",
            "conf_thresh": 0.25,
            "iou_thresh": 0.5,
        }
        model_cfg = {"imgsz": 640}

        run_detection_eval("model.pt", "data.yaml", eval_cfg, model_cfg)

        mock_yolo_cls.assert_called_once_with("model.pt")
        mock_model.val.assert_called_once_with(
            data="data.yaml",
            split="val",
            conf=0.25,
            iou=0.5,
            imgsz=640,
            verbose=False,
        )

    @patch("evaluation.detection_metrics.YOLO")
    def test_uses_defaults_when_keys_missing(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.val.return_value = MagicMock()

        run_detection_eval("model.pt", "data.yaml", {})

        call_kwargs = mock_model.val.call_args[1]
        assert call_kwargs["split"] == "val"
        assert call_kwargs["conf"] == 0.3
        assert call_kwargs["iou"] == 0.45
        assert call_kwargs["imgsz"] == 640

    @patch("evaluation.detection_metrics.YOLO")
    def test_uses_test_split(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.val.return_value = MagicMock()

        run_detection_eval("model.pt", "data.yaml", {"split": "test"})

        call_kwargs = mock_model.val.call_args[1]
        assert call_kwargs["split"] == "test"

    @patch("evaluation.detection_metrics.YOLO")
    def test_imgsz_from_eval_cfg_fallback(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.val.return_value = MagicMock()

        run_detection_eval("model.pt", "data.yaml", {"imgsz": 960}, model_cfg=None)

        call_kwargs = mock_model.val.call_args[1]
        assert call_kwargs["imgsz"] == 960

    @patch("evaluation.detection_metrics.YOLO")
    def test_returns_results(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        expected = MagicMock()
        mock_model.val.return_value = expected

        actual = run_detection_eval("model.pt", "data.yaml", {})
        assert actual is expected


class TestFormatDetectionResults:
    def test_summary_keys(self, mock_val_results):
        class_names = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        result = format_detection_results(mock_val_results, class_names)
        assert "summary" in result
        assert "per_class" in result
        for key in ("mAP50", "mAP50-95", "precision", "recall"):
            assert key in result["summary"]

    def test_summary_values(self, mock_val_results):
        class_names = {}
        result = format_detection_results(mock_val_results, class_names)
        assert result["summary"]["mAP50"] == 0.85
        assert result["summary"]["mAP50-95"] == 0.62
        assert result["summary"]["precision"] == 0.78
        assert result["summary"]["recall"] == 0.72

    def test_per_class_sorted_weakest_first(self, mock_val_results):
        class_names = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        result = format_detection_results(mock_val_results, class_names)
        per_class = result["per_class"]
        assert len(per_class) == 5
        # Weakest first: D(0.10), B(0.30), E(0.72), C(0.88), A(0.95)
        assert per_class[0]["class_name"] == "D"
        assert per_class[0]["AP50"] == pytest.approx(0.10)
        assert per_class[-1]["class_name"] == "A"
        assert per_class[-1]["AP50"] == pytest.approx(0.95)

    def test_missing_class_name_fallback(self, mock_val_results):
        result = format_detection_results(mock_val_results, {})
        # Should use fallback names like "class_0"
        names = [entry["class_name"] for entry in result["per_class"]]
        assert all(name.startswith("class_") for name in names)

    def test_per_class_has_required_fields(self, mock_val_results):
        class_names = {0: "A"}
        result = format_detection_results(mock_val_results, class_names)
        for entry in result["per_class"]:
            assert "class_id" in entry
            assert "class_name" in entry
            assert "AP50" in entry

    def test_empty_results(self):
        results = MagicMock()
        results.results_dict = {
            "metrics/mAP50(B)": 0.0,
            "metrics/mAP50-95(B)": 0.0,
            "metrics/precision(B)": 0.0,
            "metrics/recall(B)": 0.0,
        }
        results.box.ap50 = np.array([])
        results.box.ap_class_index = np.array([])

        result = format_detection_results(results, {})
        assert result["per_class"] == []
        assert result["summary"]["mAP50"] == 0.0
