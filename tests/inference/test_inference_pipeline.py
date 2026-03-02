"""Tests for the inference pipeline orchestration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from inference.run_inference import format_results, run_pipeline, run_video_inference


@pytest.fixture
def inference_cfg():
    """Create a Hydra-style DictConfig for inference."""
    return OmegaConf.create(
        {
            "model": {
                "name": "yolo11m.pt",
                "variant": "yolo11m",
                "imgsz": 640,
            },
            "inference": {
                "conf_thresh": 0.3,
                "iou_thresh": 0.45,
                "tracker": "bytetrack.yaml",
                "persist": True,
                "min_track_length": 5,
                "class_vote_method": "majority",
                "export_format": "engine",
                "half": True,
            },
            "device": 0,
        }
    )


class TestHydraInferenceConfig:
    def test_config_structure(self, inference_cfg):
        assert inference_cfg.inference.conf_thresh == 0.3
        assert inference_cfg.inference.iou_thresh == 0.45
        assert inference_cfg.inference.min_track_length == 5

    def test_model_accessible(self, inference_cfg):
        assert inference_cfg.model.name == "yolo11m.pt"
        assert inference_cfg.model.imgsz == 640

    def test_config_override(self, inference_cfg):
        """Hydra overrides work via OmegaConf.merge."""
        overridden = OmegaConf.merge(
            inference_cfg,
            {"inference": {"conf_thresh": 0.5}},
        )
        assert overridden.inference.conf_thresh == 0.5
        # Other values unchanged
        assert overridden.inference.iou_thresh == 0.45

    def test_to_container(self, inference_cfg):
        """Can convert to plain dict for passing to functions."""
        d = OmegaConf.to_container(inference_cfg.inference, resolve=True)
        assert isinstance(d, dict)
        assert d["conf_thresh"] == 0.3


class TestFormatResults:
    def test_json_serializable(self):
        counts = {0: 3, 5: 1, 10: 2}
        class_names = {0: "Advil", 5: "Aleve", 10: "Barilla"}
        result = format_results("testA_1.mp4", counts, class_names)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_structure(self):
        counts = {0: 3, 5: 1}
        class_names = {0: "Advil", 5: "Aleve"}
        result = format_results("testA_1.mp4", counts, class_names)
        assert result["video"] == "testA_1.mp4"
        assert "products" in result
        assert len(result["products"]) == 2

    def test_product_fields(self):
        counts = {0: 2}
        class_names = {0: "Advil"}
        result = format_results("test.mp4", counts, class_names)
        product = result["products"][0]
        assert "class_id" in product
        assert "class_name" in product
        assert "count" in product
        assert product["count"] == 2
        assert product["class_name"] == "Advil"

    def test_total_products(self):
        counts = {0: 3, 1: 2, 2: 5}
        result = format_results("test.mp4", counts, {})
        assert result["total_products"] == 10
        assert result["unique_classes"] == 3


def _make_mock_result(has_detections=True):
    """Create a mock YOLO result object."""
    result = MagicMock()
    result.orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    if has_detections:
        result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = [1]
        result.boxes.cls.int.return_value.cpu.return_value.tolist.return_value = [0]
        result.boxes.conf.cpu.return_value.tolist.return_value = [0.9]
    else:
        result.boxes = None

    return result


@pytest.fixture
def mock_yolo():
    """Mock YOLO model that yields two results (one with detections, one without)."""
    with patch("inference.run_inference.YOLO") as mock_cls:
        model = MagicMock()
        model.track.return_value = [
            _make_mock_result(has_detections=True),
            _make_mock_result(has_detections=False),
        ]
        mock_cls.return_value = model
        yield mock_cls


class TestVisualization:
    def test_no_visualization_by_default(self, mock_yolo, tmp_path):
        """When visualize=False, no VideoWriter is created."""
        with patch("inference.run_inference._init_video_writer") as mock_writer:
            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
            )
            mock_writer.assert_not_called()

    def test_visualize_saves_video(self, mock_yolo, tmp_path):
        """When visualize=True, VideoWriter is created and frames are written."""
        mock_writer = MagicMock()
        with patch(
            "inference.run_inference._init_video_writer", return_value=mock_writer
        ):
            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
                visualize=True,
                output_dir=tmp_path,
            )
            assert mock_writer.write.call_count == 2
            mock_writer.release.assert_called_once()

    def test_show_live_displays_frames(self, mock_yolo, tmp_path):
        """When show_live=True, cv2.imshow is called for each frame."""
        with patch("inference.run_inference.cv2") as mock_cv2:
            mock_cv2.waitKey.return_value = 0  # not 'q'
            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
                show_live=True,
            )
            assert mock_cv2.imshow.call_count == 2
            mock_cv2.destroyAllWindows.assert_called_once()

    def test_quit_key_stops_early(self, mock_yolo, tmp_path):
        """Pressing 'q' during show_live breaks the loop early."""
        with patch("inference.run_inference.cv2") as mock_cv2:
            mock_cv2.waitKey.return_value = ord("q")
            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
                show_live=True,
            )
            # Only first frame processed before 'q' breaks the loop
            assert mock_cv2.imshow.call_count == 1


class TestRunPipeline:
    """Test the full pipeline orchestration."""

    def test_processes_videos_and_writes_json(self, tmp_path):
        """run_pipeline should process videos and write result JSON files."""
        # Set up directory structure
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test_video.mp4").touch()

        output_dir = tmp_path / "outputs"

        cfg = OmegaConf.create(
            {
                "project_root": str(tmp_path),
                "data": {
                    "video_dir": "videos",
                    "output_dir": "outputs",
                },
                "inference": {
                    "conf_thresh": 0.3,
                    "iou_thresh": 0.45,
                    "tracker": "bytetrack.yaml",
                    "persist": True,
                    "min_track_length": 1,
                    "visualize": False,
                    "show_live": False,
                },
                "model": {
                    "imgsz": 640,
                },
            }
        )

        class_names = {0: "Advil", 1: "Aleve"}

        with patch("inference.run_inference.run_video_inference") as mock_infer:
            from inference.counter import TrackRecord

            track = TrackRecord(track_id=1)
            track.add_detection(0, 0.9)
            track.add_detection(0, 0.85)
            mock_infer.return_value = {1: track}

            run_pipeline(cfg, Path("model.pt"), class_names)

        result_file = output_dir / "test_video_results.json"
        assert result_file.exists()

        with open(result_file) as f:
            result = json.load(f)
        assert result["video"] == "test_video.mp4"
        assert result["total_products"] == 1

    def test_no_videos_prints_message(self, tmp_path, capsys):
        """When no MP4 files exist, should print a message and return."""
        video_dir = tmp_path / "empty_videos"
        video_dir.mkdir()

        cfg = OmegaConf.create(
            {
                "project_root": str(tmp_path),
                "data": {
                    "video_dir": "empty_videos",
                    "output_dir": "outputs",
                },
                "inference": {
                    "conf_thresh": 0.3,
                    "min_track_length": 5,
                    "visualize": False,
                    "show_live": False,
                },
                "model": {"imgsz": 640},
            }
        )

        run_pipeline(cfg, Path("model.pt"), {})
        captured = capsys.readouterr()
        assert "No MP4 files found" in captured.out
