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

    def test_list_imgsz_survives_to_container(self):
        """OmegaConf ListConfig becomes a plain Python list via to_container."""
        cfg = OmegaConf.create({"inference": {"imgsz": [384, 640]}})
        d = OmegaConf.to_container(cfg.inference, resolve=True)
        assert d["imgsz"] == [384, 640]
        assert isinstance(d["imgsz"], list)

    def test_tier1_config_defaults(self):
        """Documents the expected Tier 1 config contract."""
        cfg = OmegaConf.create(
            {
                "inference": {
                    "conf_thresh": 0.15,
                    "iou_thresh": 0.45,
                    "tracker": "conf/trackers/botsort_reid.yaml",
                    "persist": True,
                    "min_track_length": 15,
                    "class_vote_method": "majority",
                    "imgsz": [384, 640],
                    "entry_zone": {
                        "enabled": True,
                        "edge": "left",
                        "size": 0.15,
                    },
                }
            }
        )
        d = OmegaConf.to_container(cfg.inference, resolve=True)
        assert d["conf_thresh"] == 0.15
        assert d["min_track_length"] == 15
        assert d["imgsz"] == [384, 640]
        assert d["entry_zone"]["enabled"] is True
        assert d["tracker"] == "conf/trackers/botsort_reid.yaml"

    def test_tier2_stitch_config_defaults(self):
        """Documents the expected Tier 2 stitch config contract."""
        cfg = OmegaConf.create(
            {
                "inference": {
                    "stitch": {
                        "max_gap": 90,
                        "max_distance": 0.15,
                    },
                }
            }
        )
        d = OmegaConf.to_container(cfg.inference, resolve=True)
        assert d["stitch"]["max_gap"] == 90
        assert d["stitch"]["max_distance"] == 0.15


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


def _make_mock_result(has_detections=True, track_ids=None, centers=None):
    """Create a mock YOLO result object.

    Args:
        has_detections: Whether the result has detections.
        track_ids: List of track IDs. Defaults to [1].
        centers: List of [cx, cy] normalized centers. Defaults to [[0.5, 0.5]].
    """
    result = MagicMock()
    result.orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    if has_detections:
        if track_ids is None:
            track_ids = [1]
        if centers is None:
            centers = [[0.5, 0.5]] * len(track_ids)

        n = len(track_ids)
        result.boxes.id.int.return_value.cpu.return_value.tolist.return_value = (
            track_ids
        )
        result.boxes.cls.int.return_value.cpu.return_value.tolist.return_value = [0] * n
        result.boxes.conf.cpu.return_value.tolist.return_value = [0.9] * n

        # xywhn[:, :2] returns normalized centers
        xywhn_mock = MagicMock()
        xywhn_mock.__getitem__ = lambda self, idx: MagicMock(
            cpu=lambda: MagicMock(tolist=lambda: centers)
        )
        result.boxes.xywhn = xywhn_mock
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

    def test_entry_zone_filters_when_enabled(self, tmp_path):
        """When entry_zone is enabled, only edge-entry tracks are counted."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test_video.mp4").touch()
        output_dir = tmp_path / "outputs"

        cfg = OmegaConf.create(
            {
                "project_root": str(tmp_path),
                "data": {"video_dir": "videos", "output_dir": "outputs"},
                "inference": {
                    "conf_thresh": 0.3,
                    "iou_thresh": 0.45,
                    "tracker": "bytetrack.yaml",
                    "persist": True,
                    "min_track_length": 1,
                    "visualize": False,
                    "show_live": False,
                    "entry_zone": {"enabled": True, "edge": "left", "size": 0.15},
                },
                "model": {"imgsz": 640},
            }
        )

        with patch("inference.run_inference.run_video_inference") as mock_infer:
            from inference.counter import TrackRecord

            # Track 1: entered from left edge (should be kept)
            t1 = TrackRecord(track_id=1, entry_position=(0.05, 0.5))
            t1.add_detection(0, 0.9)
            # Track 2: appeared in center (should be filtered)
            t2 = TrackRecord(track_id=2, entry_position=(0.5, 0.5))
            t2.add_detection(1, 0.9)
            mock_infer.return_value = {1: t1, 2: t2}

            run_pipeline(cfg, Path("model.pt"), {0: "Advil", 1: "Aleve"})

        result_file = output_dir / "test_video_results.json"
        with open(result_file) as f:
            result = json.load(f)
        assert result["total_products"] == 1

    def test_stitch_merges_before_filter_short(self, tmp_path):
        """Two 8-detection fragments stitch to 16 and survive min_track_length=15."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test_video.mp4").touch()
        output_dir = tmp_path / "outputs"

        cfg = OmegaConf.create(
            {
                "project_root": str(tmp_path),
                "data": {"video_dir": "videos", "output_dir": "outputs"},
                "inference": {
                    "conf_thresh": 0.3,
                    "iou_thresh": 0.45,
                    "tracker": "bytetrack.yaml",
                    "persist": True,
                    "min_track_length": 15,
                    "visualize": False,
                    "show_live": False,
                    "stitch": {"max_gap": 90, "max_distance": 0.15},
                },
                "model": {"imgsz": 640},
            }
        )

        with patch("inference.run_inference.run_video_inference") as mock_infer:
            from inference.counter import TrackRecord

            t1 = TrackRecord(
                track_id=1,
                detections=[(0, 0.9)] * 8,
                entry_position=(0.05, 0.5),
                first_frame=0,
                last_frame=40,
                last_position=(0.3, 0.5),
            )
            t2 = TrackRecord(
                track_id=2,
                detections=[(0, 0.9)] * 8,
                entry_position=(0.35, 0.5),
                first_frame=50,
                last_frame=90,
                last_position=(0.5, 0.5),
            )
            mock_infer.return_value = {1: t1, 2: t2}

            run_pipeline(cfg, Path("model.pt"), {0: "Advil"})

        result_file = output_dir / "test_video_results.json"
        with open(result_file) as f:
            result = json.load(f)
        assert result["total_products"] == 1

    def test_stitch_disabled_when_max_gap_zero(self, tmp_path):
        """Same fragments with max_gap=0: both filtered, total=0."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test_video.mp4").touch()
        output_dir = tmp_path / "outputs"

        cfg = OmegaConf.create(
            {
                "project_root": str(tmp_path),
                "data": {"video_dir": "videos", "output_dir": "outputs"},
                "inference": {
                    "conf_thresh": 0.3,
                    "iou_thresh": 0.45,
                    "tracker": "bytetrack.yaml",
                    "persist": True,
                    "min_track_length": 15,
                    "visualize": False,
                    "show_live": False,
                    "stitch": {"max_gap": 0, "max_distance": 0.15},
                },
                "model": {"imgsz": 640},
            }
        )

        with patch("inference.run_inference.run_video_inference") as mock_infer:
            from inference.counter import TrackRecord

            t1 = TrackRecord(
                track_id=1,
                detections=[(0, 0.9)] * 8,
                entry_position=(0.05, 0.5),
                first_frame=0,
                last_frame=40,
                last_position=(0.3, 0.5),
            )
            t2 = TrackRecord(
                track_id=2,
                detections=[(0, 0.9)] * 8,
                entry_position=(0.35, 0.5),
                first_frame=50,
                last_frame=90,
                last_position=(0.5, 0.5),
            )
            mock_infer.return_value = {1: t1, 2: t2}

            run_pipeline(cfg, Path("model.pt"), {0: "Advil"})

        result_file = output_dir / "test_video_results.json"
        with open(result_file) as f:
            result = json.load(f)
        assert result["total_products"] == 0

    def test_entry_zone_passthrough_when_disabled(self, tmp_path):
        """When entry_zone is disabled, all tracks pass through."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test_video.mp4").touch()
        output_dir = tmp_path / "outputs"

        cfg = OmegaConf.create(
            {
                "project_root": str(tmp_path),
                "data": {"video_dir": "videos", "output_dir": "outputs"},
                "inference": {
                    "conf_thresh": 0.3,
                    "iou_thresh": 0.45,
                    "tracker": "bytetrack.yaml",
                    "persist": True,
                    "min_track_length": 1,
                    "visualize": False,
                    "show_live": False,
                    "entry_zone": {"enabled": False, "edge": "left", "size": 0.15},
                },
                "model": {"imgsz": 640},
            }
        )

        with patch("inference.run_inference.run_video_inference") as mock_infer:
            from inference.counter import TrackRecord

            # Track in center — would be filtered if entry zone were enabled
            t1 = TrackRecord(track_id=1, entry_position=(0.5, 0.5))
            t1.add_detection(0, 0.9)
            mock_infer.return_value = {1: t1}

            run_pipeline(cfg, Path("model.pt"), {0: "Advil"})

        result_file = output_dir / "test_video_results.json"
        with open(result_file) as f:
            result = json.load(f)
        assert result["total_products"] == 1


class TestImgszPassthrough:
    """Verify that imgsz is correctly forwarded to model.track()."""

    def test_scalar_imgsz_passed_to_model_track(self):
        """Scalar imgsz from model_cfg is forwarded to model.track()."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [_make_mock_result(has_detections=False)]
            mock_cls.return_value = model

            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
                model_cfg={"imgsz": 640},
            )

            assert model.track.call_args[1]["imgsz"] == 640

    def test_rectangular_imgsz_from_model_cfg_passed(self):
        """List imgsz [384, 640] from model_cfg is forwarded as-is."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [_make_mock_result(has_detections=False)]
            mock_cls.return_value = model

            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
                model_cfg={"imgsz": [384, 640]},
            )

            assert model.track.call_args[1]["imgsz"] == [384, 640]

    def test_inference_cfg_imgsz_overrides_model_cfg(self):
        """inference_cfg imgsz should take precedence over model_cfg imgsz."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [_make_mock_result(has_detections=False)]
            mock_cls.return_value = model

            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {"imgsz": [384, 640]},
                model_cfg={"imgsz": 640},
            )

            assert model.track.call_args[1]["imgsz"] == [384, 640]

    def test_imgsz_fallback_to_model_cfg(self):
        """When inference_cfg has no imgsz, model_cfg imgsz is used."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [_make_mock_result(has_detections=False)]
            mock_cls.return_value = model

            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
                model_cfg={"imgsz": 640},
            )

            assert model.track.call_args[1]["imgsz"] == 640

    def test_imgsz_default_when_not_in_any_cfg(self):
        """When neither config has imgsz, defaults to 640."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [_make_mock_result(has_detections=False)]
            mock_cls.return_value = model

            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {},
            )

            assert model.track.call_args[1]["imgsz"] == 640


class TestConfigPropagation:
    """Verify that inference config params reach model.track()."""

    def test_conf_thresh_forwarded(self):
        """conf_thresh is forwarded as 'conf' to model.track()."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [_make_mock_result(has_detections=False)]
            mock_cls.return_value = model

            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {"conf_thresh": 0.15},
            )

            assert model.track.call_args[1]["conf"] == 0.15

    def test_tracker_path_forwarded(self):
        """Custom tracker path is forwarded to model.track()."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [_make_mock_result(has_detections=False)]
            mock_cls.return_value = model

            run_video_inference(
                Path("model.pt"),
                Path("video.mp4"),
                {"tracker": "conf/trackers/botsort_reid.yaml"},
            )

            assert (
                model.track.call_args[1]["tracker"] == "conf/trackers/botsort_reid.yaml"
            )


class TestFrameTracking:
    """Verify that run_video_inference sets first_frame, last_frame, and last_position."""

    def test_first_and_last_frame_set(self):
        """Track seen in 3 frames gets first_frame=0, last_frame=2."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.3, 0.4]]
                ),
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.4, 0.5]]
                ),
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.5, 0.6]]
                ),
            ]
            mock_cls.return_value = model

            tracks = run_video_inference(Path("model.pt"), Path("video.mp4"), {})

        assert tracks[1].first_frame == 0
        assert tracks[1].last_frame == 2

    def test_last_position_updates(self):
        """Second detection at (0.6,0.7) sets last_position=(0.6,0.7)."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.3, 0.4]]
                ),
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.6, 0.7]]
                ),
            ]
            mock_cls.return_value = model

            tracks = run_video_inference(Path("model.pt"), Path("video.mp4"), {})

        assert tracks[1].last_position == pytest.approx((0.6, 0.7))

    def test_frame_index_includes_empty_frames(self):
        """Empty frames (no detections) still count toward frame_idx."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.3, 0.4]]
                ),
                _make_mock_result(has_detections=False),
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.5, 0.6]]
                ),
            ]
            mock_cls.return_value = model

            tracks = run_video_inference(Path("model.pt"), Path("video.mp4"), {})

        assert tracks[1].first_frame == 0
        assert tracks[1].last_frame == 2  # frame 1 was empty but still counted


class TestEntryPositionCapture:
    """Verify that run_video_inference captures entry_position on tracks."""

    def test_entry_position_set_on_first_detection(self):
        """entry_position should be the normalized center of the first bbox."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.1, 0.2]]
                ),
                _make_mock_result(
                    has_detections=True, track_ids=[1], centers=[[0.3, 0.4]]
                ),
            ]
            mock_cls.return_value = model

            tracks = run_video_inference(Path("model.pt"), Path("video.mp4"), {})

        assert 1 in tracks
        # entry_position should be from the first detection, not overwritten
        assert tracks[1].entry_position == pytest.approx((0.1, 0.2))
        assert tracks[1].length == 2

    def test_multiple_tracks_each_get_entry_position(self):
        """Each track gets its own entry_position from its first appearance."""
        with patch("inference.run_inference.YOLO") as mock_cls:
            model = MagicMock()
            model.track.return_value = [
                _make_mock_result(
                    has_detections=True,
                    track_ids=[1, 2],
                    centers=[[0.1, 0.2], [0.8, 0.9]],
                ),
            ]
            mock_cls.return_value = model

            tracks = run_video_inference(Path("model.pt"), Path("video.mp4"), {})

        assert tracks[1].entry_position == pytest.approx((0.1, 0.2))
        assert tracks[2].entry_position == pytest.approx((0.8, 0.9))
