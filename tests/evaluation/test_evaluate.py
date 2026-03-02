"""Tests for the evaluation entry point."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from evaluation.evaluate import run_evaluation


@pytest.fixture
def base_cfg(tmp_path):
    """Create a full Hydra-style DictConfig for evaluation."""
    return OmegaConf.create(
        {
            "project_root": str(tmp_path),
            "model": {
                "name": "yolo11m.pt",
                "variant": "yolo11m",
                "imgsz": 640,
            },
            "training": {
                "project": "runs/detect",
                "name": "checkout_v1",
            },
            "data": {
                "dataset_dir": "datasets/checkout",
                "label_file": "data/label.txt",
            },
            "evaluation": {
                "mode": "detection",
                "split": "val",
                "conf_thresh": 0.3,
                "iou_thresh": 0.45,
                "ground_truth_dir": None,
                "predictions_dir": None,
                "save_results": True,
                "output_dir": "outputs/eval",
            },
        }
    )


def _make_mock_val_results():
    """Create a mock YOLO validation results object."""
    results = MagicMock()
    results.results_dict = {
        "metrics/mAP50(B)": 0.80,
        "metrics/mAP50-95(B)": 0.55,
        "metrics/precision(B)": 0.75,
        "metrics/recall(B)": 0.70,
    }
    results.box.ap50 = np.array([0.90, 0.40])
    results.box.ap_class_index = np.array([0, 1])
    return results


class TestEvaluationConfig:
    def test_config_structure(self, base_cfg):
        assert base_cfg.evaluation.mode == "detection"
        assert base_cfg.evaluation.split == "val"
        assert base_cfg.evaluation.conf_thresh == 0.3
        assert base_cfg.evaluation.iou_thresh == 0.45

    def test_config_override(self, base_cfg):
        overridden = OmegaConf.merge(
            base_cfg,
            {"evaluation": {"conf_thresh": 0.5, "split": "test"}},
        )
        assert overridden.evaluation.conf_thresh == 0.5
        assert overridden.evaluation.split == "test"
        # Other values unchanged
        assert overridden.evaluation.iou_thresh == 0.45


class TestRunEvaluationDetection:
    @patch("evaluation.evaluate.run_detection_eval")
    @patch("evaluation.evaluate.parse_label_file")
    def test_detection_mode(self, mock_parse, mock_eval, base_cfg, tmp_path):
        mock_parse.return_value = {1: "Advil", 2: "Aleve"}
        mock_eval.return_value = _make_mock_val_results()

        results = run_evaluation(base_cfg)

        assert "detection" in results
        assert "summary" in results["detection"]
        assert results["detection"]["summary"]["mAP50"] == 0.80

        # Verify JSON was saved
        output_file = tmp_path / "outputs" / "eval" / "eval_results.json"
        assert output_file.exists()
        with open(output_file) as f:
            saved = json.load(f)
        assert saved["detection"]["summary"]["mAP50"] == 0.80

    @patch("evaluation.evaluate.run_detection_eval")
    @patch("evaluation.evaluate.parse_label_file")
    def test_passes_eval_config_to_val(self, mock_parse, mock_eval, base_cfg):
        mock_parse.return_value = {}
        mock_eval.return_value = _make_mock_val_results()

        base_cfg.evaluation.conf_thresh = 0.5
        base_cfg.evaluation.split = "test"
        run_evaluation(base_cfg)

        call_args = mock_eval.call_args
        eval_cfg = call_args[0][2]
        assert eval_cfg["conf_thresh"] == 0.5
        assert eval_cfg["split"] == "test"


class TestRunEvaluationCounting:
    def test_counting_mode(self, base_cfg, tmp_path):
        # Set up counting dirs
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        # Write test data
        pred_data = {"products": [{"class_id": 0, "count": 3}]}
        gt_data = {"products": [{"class_id": 0, "count": 3}]}
        (pred_dir / "vid1_results.json").write_text(json.dumps(pred_data))
        (gt_dir / "vid1_gt.json").write_text(json.dumps(gt_data))

        base_cfg.evaluation.mode = "counting"
        base_cfg.evaluation.predictions_dir = "preds"
        base_cfg.evaluation.ground_truth_dir = "gt"

        results = run_evaluation(base_cfg)
        assert "counting" in results
        assert results["counting"]["aggregate"]["mae"] == 0.0

    def test_counting_skipped_without_dirs(self, base_cfg, capsys):
        base_cfg.evaluation.mode = "counting"
        results = run_evaluation(base_cfg)
        assert "counting" not in results
        captured = capsys.readouterr()
        assert "ground_truth_dir and predictions_dir must be set" in captured.out


class TestRunEvaluationBothMode:
    @patch("evaluation.evaluate.run_detection_eval")
    @patch("evaluation.evaluate.parse_label_file")
    def test_both_mode(self, mock_parse, mock_eval, base_cfg, tmp_path):
        mock_parse.return_value = {1: "Advil"}
        mock_eval.return_value = _make_mock_val_results()

        # Set up counting dirs
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()
        pred_data = {"products": [{"class_id": 0, "count": 2}]}
        gt_data = {"products": [{"class_id": 0, "count": 2}]}
        (pred_dir / "v1_results.json").write_text(json.dumps(pred_data))
        (gt_dir / "v1_gt.json").write_text(json.dumps(gt_data))

        base_cfg.evaluation.mode = "both"
        base_cfg.evaluation.predictions_dir = "preds"
        base_cfg.evaluation.ground_truth_dir = "gt"

        results = run_evaluation(base_cfg)
        assert "detection" in results
        assert "counting" in results


class TestSaveResults:
    @patch("evaluation.evaluate.run_detection_eval")
    @patch("evaluation.evaluate.parse_label_file")
    def test_save_disabled(self, mock_parse, mock_eval, base_cfg, tmp_path):
        mock_parse.return_value = {}
        mock_eval.return_value = _make_mock_val_results()
        base_cfg.evaluation.save_results = False

        run_evaluation(base_cfg)

        output_file = tmp_path / "outputs" / "eval" / "eval_results.json"
        assert not output_file.exists()
