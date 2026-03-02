"""Tests for training configuration and setup with Hydra/OmegaConf."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from training.train_classifier import (
    _resolve_last_checkpoint as _resolve_last_checkpoint_cls,
)
from training.train_classifier import build_classify_args, train_classifier
from training.train_detector import (
    _resolve_last_checkpoint,
    build_train_args,
    get_model_name,
    train,
)


@pytest.fixture
def detect_cfg():
    """Create a full Hydra-style DictConfig for detection training."""
    return OmegaConf.create(
        {
            "model": {
                "name": "yolo11m.pt",
                "variant": "yolo11m",
                "imgsz": 640,
            },
            "training": {
                "epochs": 100,
                "batch": 16,
                "patience": 20,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "lrf": 0.01,
                "warmup_epochs": 5,
                "cos_lr": True,
                "hsv_h": 0.015,
                "hsv_s": 0.5,
                "hsv_v": 0.3,
                "degrees": 15,
                "translate": 0.1,
                "scale": 0.5,
                "flipud": 0.1,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.1,
                "copy_paste": 0.1,
                "project": "runs/detect",
                "name": "checkout_v1",
                "exist_ok": True,
            },
            "device": 0,
        }
    )


@pytest.fixture
def classify_cfg():
    """Create a Hydra-style DictConfig for classification training."""
    return OmegaConf.create(
        {
            "model": {
                "name": "yolo11m-cls.pt",
                "variant": "yolo11m-cls",
                "imgsz": 224,
            },
            "training": {
                "epochs": 50,
                "batch": 64,
                "patience": 15,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "project": "runs/classify",
                "name": "pretrain_v1",
                "exist_ok": True,
            },
            "device": 0,
        }
    )


class TestGetModelName:
    def test_extracts_model_name(self, detect_cfg):
        assert get_model_name(detect_cfg) == "yolo11m.pt"

    def test_default_model(self, detect_cfg):
        """Switching model is just a config override."""
        cfg2 = OmegaConf.merge(detect_cfg, {"model": {"name": "yolo11l.pt"}})
        assert get_model_name(cfg2) == "yolo11l.pt"


class TestBuildTrainArgs:
    def test_returns_dict(self, detect_cfg):
        args = build_train_args(detect_cfg, "/tmp/data.yaml")
        assert isinstance(args, dict)

    def test_includes_data_path(self, detect_cfg):
        args = build_train_args(detect_cfg, "/tmp/data.yaml")
        assert args["data"] == "/tmp/data.yaml"

    def test_excludes_model_keys(self, detect_cfg):
        """Model variant should not be in training args."""
        args = build_train_args(detect_cfg, "/tmp/data.yaml")
        assert "variant" not in args
        assert args["name"] == "checkout_v1"  # YOLO run name is passed through

    def test_includes_imgsz_from_model(self, detect_cfg):
        args = build_train_args(detect_cfg, "/tmp/data.yaml")
        assert args["imgsz"] == 640

    def test_sets_device(self, detect_cfg):
        args = build_train_args(detect_cfg, "/tmp/data.yaml", device=1)
        assert args["device"] == 1

    def test_preserves_augmentation_params(self, detect_cfg):
        args = build_train_args(detect_cfg, "/tmp/data.yaml")
        assert args["mosaic"] == 1.0
        assert args["mixup"] == 0.1
        assert args["degrees"] == 15

    def test_preserves_training_params(self, detect_cfg):
        args = build_train_args(detect_cfg, "/tmp/data.yaml")
        assert args["epochs"] == 100
        assert args["batch"] == 16
        assert args["optimizer"] == "AdamW"

    def test_model_override_changes_imgsz(self):
        """Switching to a different model config should change imgsz."""
        cfg = OmegaConf.create(
            {
                "model": {"name": "yolo11l.pt", "variant": "yolo11l", "imgsz": 960},
                "training": {"epochs": 50, "batch": 8},
            }
        )
        args = build_train_args(cfg, "/tmp/data.yaml")
        assert args["imgsz"] == 960

    def test_seed_passed_through(self):
        """Seed must reach YOLO.train() for reproducible training (B1)."""
        cfg = OmegaConf.create(
            {
                "model": {"name": "yolo11m.pt", "variant": "yolo11m", "imgsz": 640},
                "training": {"epochs": 10, "batch": 16, "seed": 42},
            }
        )
        args = build_train_args(cfg, "/tmp/data.yaml")
        assert args["seed"] == 42


class TestBuildClassifyArgs:
    def test_returns_dict(self, classify_cfg):
        args = build_classify_args(classify_cfg, "/path/to/data")
        assert isinstance(args, dict)

    def test_includes_data(self, classify_cfg):
        args = build_classify_args(classify_cfg, "/path/to/data")
        assert args["data"] == "/path/to/data"

    def test_excludes_model_keys(self, classify_cfg):
        args = build_classify_args(classify_cfg, "/path/to/data")
        assert "variant" not in args
        assert args["name"] == "pretrain_v1"  # YOLO run name is passed through

    def test_uses_cls_imgsz(self, classify_cfg):
        args = build_classify_args(classify_cfg, "/path/to/data")
        assert args["imgsz"] == 224

    def test_no_training_key_defaults_to_empty(self):
        """B3: DictConfig without 'training' key should not leak non-training keys."""
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "yolo11m-cls.pt",
                    "variant": "yolo11m-cls",
                    "imgsz": 224,
                },
                "seed": 42,
                "project_root": "/some/path",
                "device": 0,
            }
        )
        args = build_classify_args(cfg, "/path/to/data")
        # Only data, device, and imgsz should be present — no leaked top-level keys
        assert "seed" not in args
        assert "project_root" not in args
        assert args["data"] == "/path/to/data"
        assert args["device"] == 0


class TestTrainClassifierProjectPath:
    """B2: train_classifier must absolutify project path like train_detector does."""

    @patch("training.train_classifier.YOLO")
    def test_project_path_is_absolute(self, mock_yolo_cls, tmp_path):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        project_root = str(tmp_path / "project")
        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "yolo11m-cls.pt",
                    "variant": "yolo11m-cls",
                    "imgsz": 224,
                },
                "training": {
                    "epochs": 5,
                    "batch": 32,
                    "project": "runs/classify",
                    "name": "test_v1",
                    "exist_ok": True,
                },
                "device": 0,
                "project_root": project_root,
            }
        )

        train_classifier(cfg, str(tmp_path / "data"))
        call_kwargs = mock_model.train.call_args[1]
        assert Path(call_kwargs["project"]) == Path(project_root) / "runs" / "classify"


class TestTrainDetectorProjectPath:
    """Verify train_detector absolutifies project path (existing behavior)."""

    @patch("training.train_detector.YOLO")
    def test_project_path_is_absolute(self, mock_yolo_cls, tmp_path):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        project_root = str(tmp_path / "project")
        cfg = OmegaConf.create(
            {
                "model": {"name": "yolo11m.pt", "variant": "yolo11m", "imgsz": 640},
                "training": {
                    "epochs": 5,
                    "batch": 16,
                    "project": "runs/detect",
                    "name": "test_v1",
                    "exist_ok": True,
                },
                "device": 0,
                "project_root": project_root,
            }
        )

        train(cfg, str(tmp_path / "data.yaml"))
        call_kwargs = mock_model.train.call_args[1]
        assert Path(call_kwargs["project"]) == Path(project_root) / "runs" / "detect"


class TestResumeDetector:
    """Test resume training support for detection."""

    def _make_cfg(self, tmp_path, resume=False):
        project_root = str(tmp_path / "project")
        return OmegaConf.create(
            {
                "model": {"name": "yolo11m.pt", "variant": "yolo11m", "imgsz": 640},
                "training": {
                    "epochs": 5,
                    "batch": 16,
                    "project": "runs/detect",
                    "name": "test_v1",
                    "exist_ok": True,
                    "resume": resume,
                },
                "device": 0,
                "project_root": project_root,
            }
        )

    @patch("training.train_detector.YOLO")
    def test_resume_loads_last_checkpoint(self, mock_yolo_cls, tmp_path):
        cfg = self._make_cfg(tmp_path, resume=True)
        # Create the checkpoint file
        checkpoint = (
            tmp_path / "project" / "runs" / "detect" / "test_v1" / "weights" / "last.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        train(cfg, str(tmp_path / "data.yaml"))
        mock_yolo_cls.assert_called_once_with(str(checkpoint))

    @patch("training.train_detector.YOLO")
    def test_resume_passes_resume_to_yolo_train(self, mock_yolo_cls, tmp_path):
        cfg = self._make_cfg(tmp_path, resume=True)
        checkpoint = (
            tmp_path / "project" / "runs" / "detect" / "test_v1" / "weights" / "last.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        train(cfg, str(tmp_path / "data.yaml"))
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["resume"] is True

    def test_resume_raises_if_no_checkpoint(self, tmp_path):
        cfg = self._make_cfg(tmp_path, resume=True)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            _resolve_last_checkpoint(cfg)

    @patch("training.train_detector.YOLO")
    def test_default_uses_pretrained_model(self, mock_yolo_cls, tmp_path):
        cfg = self._make_cfg(tmp_path, resume=False)
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        train(cfg, str(tmp_path / "data.yaml"))
        expected = str(Path(cfg.project_root) / "models" / "yolo11m.pt")
        mock_yolo_cls.assert_called_once_with(expected)


class TestResumeClassifier:
    """Test resume training support for classification."""

    def _make_cfg(self, tmp_path, resume=False):
        project_root = str(tmp_path / "project")
        return OmegaConf.create(
            {
                "model": {
                    "name": "yolo11m-cls.pt",
                    "variant": "yolo11m-cls",
                    "imgsz": 224,
                },
                "training": {
                    "epochs": 5,
                    "batch": 32,
                    "project": "runs/classify",
                    "name": "test_v1",
                    "exist_ok": True,
                    "resume": resume,
                },
                "device": 0,
                "project_root": project_root,
            }
        )

    @patch("training.train_classifier.YOLO")
    def test_resume_loads_last_checkpoint(self, mock_yolo_cls, tmp_path):
        cfg = self._make_cfg(tmp_path, resume=True)
        checkpoint = (
            tmp_path
            / "project"
            / "runs"
            / "classify"
            / "test_v1"
            / "weights"
            / "last.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        train_classifier(cfg, str(tmp_path / "data"))
        mock_yolo_cls.assert_called_once_with(str(checkpoint))

    def test_resume_raises_if_no_checkpoint(self, tmp_path):
        cfg = self._make_cfg(tmp_path, resume=True)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            _resolve_last_checkpoint_cls(cfg)

    @patch("training.train_classifier.YOLO")
    def test_resume_passes_resume_to_yolo_train(self, mock_yolo_cls, tmp_path):
        cfg = self._make_cfg(tmp_path, resume=True)
        checkpoint = (
            tmp_path
            / "project"
            / "runs"
            / "classify"
            / "test_v1"
            / "weights"
            / "last.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        train_classifier(cfg, str(tmp_path / "data"))
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["resume"] is True


class TestBuildTrainArgsResume:
    """Test that resume flag flows through build_train_args."""

    def test_resume_flows_through(self):
        cfg = OmegaConf.create(
            {
                "model": {"name": "yolo11m.pt", "variant": "yolo11m", "imgsz": 640},
                "training": {"epochs": 10, "batch": 16, "resume": True},
            }
        )
        args = build_train_args(cfg, "/tmp/data.yaml")
        assert args["resume"] is True

    def test_resume_false_flows_through(self):
        cfg = OmegaConf.create(
            {
                "model": {"name": "yolo11m.pt", "variant": "yolo11m", "imgsz": 640},
                "training": {"epochs": 10, "batch": 16, "resume": False},
            }
        )
        args = build_train_args(cfg, "/tmp/data.yaml")
        assert args["resume"] is False
