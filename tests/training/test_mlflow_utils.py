"""Tests for MLflow tracking integration."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from training.mlflow_utils import (
    _ENV_KEYS,
    _log_hydra_config,
    _resolve_experiment_name,
    _resolve_tracking_uri,
    mlflow_context,
)


@pytest.fixture
def base_cfg(tmp_path):
    """Base config with mlflow enabled."""
    return OmegaConf.create(
        {
            "model": {"name": "yolo11m.pt", "variant": "yolo11m", "imgsz": 640},
            "training": {
                "epochs": 10,
                "batch": 16,
                "name": "checkout_v1",
            },
            "data": {
                "dataset_dir": "datasets/checkout",
                "label_file": "data/label.txt",
                "num_scenes": 30000,
                "val_ratio": 0.15,
            },
            "mlflow": {
                "enabled": True,
                "tracking_uri": "runs/mlflow",
                "experiment_name": None,
                "run_name": None,
            },
            "device": 0,
            "seed": 42,
            "project_root": str(tmp_path),
        }
    )


@pytest.fixture
def disabled_cfg(base_cfg):
    """Config with mlflow disabled."""
    return OmegaConf.merge(base_cfg, {"mlflow": {"enabled": False}})


@pytest.fixture
def no_mlflow_cfg(tmp_path):
    """Config without mlflow section at all."""
    return OmegaConf.create(
        {
            "model": {"name": "yolo11m.pt", "variant": "yolo11m", "imgsz": 640},
            "training": {"epochs": 10, "batch": 16, "name": "test_run"},
            "device": 0,
            "project_root": str(tmp_path),
        }
    )


# --- _resolve_tracking_uri ---


class TestResolveTrackingUri:
    def test_relative_path_resolved_to_file_uri(self, base_cfg, tmp_path):
        uri = _resolve_tracking_uri(base_cfg)
        expected = (tmp_path / "runs" / "mlflow").as_uri()
        assert uri == expected

    def test_absolute_path_converted_to_file_uri(self, base_cfg, tmp_path):
        abs_path = str(tmp_path / "custom" / "mlflow")
        cfg = OmegaConf.merge(base_cfg, {"mlflow": {"tracking_uri": abs_path}})
        expected = (tmp_path / "custom" / "mlflow").as_uri()
        assert _resolve_tracking_uri(cfg) == expected

    def test_null_uri_defaults_to_runs_mlflow(self, base_cfg, tmp_path):
        cfg = OmegaConf.merge(base_cfg, {"mlflow": {"tracking_uri": None}})
        uri = _resolve_tracking_uri(cfg)
        expected = (tmp_path / "runs" / "mlflow").as_uri()
        assert uri == expected

    def test_uri_scheme_preserved(self, base_cfg):
        cfg = OmegaConf.merge(
            base_cfg, {"mlflow": {"tracking_uri": "http://localhost:5000"}}
        )
        assert _resolve_tracking_uri(cfg) == "http://localhost:5000"


# --- _resolve_experiment_name ---


class TestResolveExperimentName:
    def test_null_falls_back_to_training_name(self, base_cfg):
        name = _resolve_experiment_name(base_cfg)
        assert name == "checkout_v1"

    def test_explicit_name_used(self, base_cfg):
        cfg = OmegaConf.merge(
            base_cfg, {"mlflow": {"experiment_name": "my_experiment"}}
        )
        assert _resolve_experiment_name(cfg) == "my_experiment"

    def test_no_training_name_defaults(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "mlflow": {
                    "enabled": True,
                    "tracking_uri": "runs/mlflow",
                    "experiment_name": None,
                },
                "training": {"epochs": 10},
                "project_root": str(tmp_path),
            }
        )
        assert _resolve_experiment_name(cfg) == "Default"


# --- _log_hydra_config ---


class TestLogHydraConfig:
    @patch("training.mlflow_utils.mlflow")
    def test_logs_model_params(self, mock_mlflow, base_cfg):
        _log_hydra_config(base_cfg)
        logged = mock_mlflow.log_params.call_args[0][0]
        assert logged["model.variant"] == "yolo11m"
        assert logged["model.imgsz"] == 640

    @patch("training.mlflow_utils.mlflow")
    def test_logs_data_params(self, mock_mlflow, base_cfg):
        _log_hydra_config(base_cfg)
        logged = mock_mlflow.log_params.call_args[0][0]
        assert logged["data.dataset_dir"] == "datasets/checkout"
        assert logged["data.num_scenes"] == 30000

    @patch("training.mlflow_utils.mlflow")
    def test_logs_global_settings(self, mock_mlflow, base_cfg):
        _log_hydra_config(base_cfg)
        logged = mock_mlflow.log_params.call_args[0][0]
        assert logged["device"] == 0
        assert logged["seed"] == 42

    @patch("training.mlflow_utils.mlflow")
    def test_no_params_skips_log(self, mock_mlflow, tmp_path):
        cfg = OmegaConf.create({"project_root": str(tmp_path)})
        _log_hydra_config(cfg)
        mock_mlflow.log_params.assert_not_called()


# --- mlflow_context ---


class TestMlflowContextDisabled:
    def test_noop_when_disabled(self, disabled_cfg):
        """Should yield immediately without touching env vars."""
        with mlflow_context(disabled_cfg):
            assert os.environ.get("MLFLOW_TRACKING_URI") is None or os.environ.get(
                "MLFLOW_TRACKING_URI"
            ) != str(Path(disabled_cfg.project_root) / "runs" / "mlflow")

    def test_noop_when_no_mlflow_section(self, no_mlflow_cfg):
        """Should yield immediately when mlflow section is missing."""
        with mlflow_context(no_mlflow_cfg):
            pass  # Should not raise


class TestMlflowContextNotInstalled:
    @patch("training.mlflow_utils._is_mlflow_available", return_value=False)
    def test_noop_when_mlflow_not_installed(self, mock_avail, base_cfg):
        """Should gracefully degrade when mlflow is not installed."""
        with mlflow_context(base_cfg):
            pass  # Should not raise
        mock_avail.assert_called_once()


class TestMlflowContextEnabled:
    @patch("training.mlflow_utils._is_mlflow_available", return_value=True)
    @patch("training.mlflow_utils.mlflow")
    def test_sets_env_vars(self, mock_mlflow, mock_avail, base_cfg, tmp_path):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with mlflow_context(base_cfg):
            assert (
                os.environ["MLFLOW_TRACKING_URI"]
                == (tmp_path / "runs" / "mlflow").as_uri()
            )
            assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "checkout_v1"
            assert os.environ["MLFLOW_RUN"] == "test-run-123"

    @patch("training.mlflow_utils._is_mlflow_available", return_value=True)
    @patch("training.mlflow_utils.mlflow")
    def test_restores_env_vars(self, mock_mlflow, mock_avail, base_cfg):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-456"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Set a pre-existing value
        os.environ["MLFLOW_TRACKING_URI"] = "original-uri"

        try:
            with mlflow_context(base_cfg):
                assert os.environ["MLFLOW_TRACKING_URI"] != "original-uri"

            # After context, original value should be restored
            assert os.environ["MLFLOW_TRACKING_URI"] == "original-uri"
        finally:
            os.environ.pop("MLFLOW_TRACKING_URI", None)

    @patch("training.mlflow_utils._is_mlflow_available", return_value=True)
    @patch("training.mlflow_utils.mlflow")
    def test_cleans_env_vars_when_none_before(self, mock_mlflow, mock_avail, base_cfg):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-789"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Ensure env vars are not set before
        for key in _ENV_KEYS:
            os.environ.pop(key, None)

        with mlflow_context(base_cfg):
            pass

        # After context, env vars should be cleaned up
        for key in _ENV_KEYS:
            assert key not in os.environ, f"{key} was not cleaned up"

    @patch("training.mlflow_utils._is_mlflow_available", return_value=True)
    @patch("training.mlflow_utils.mlflow")
    def test_calls_set_tracking_uri_and_experiment(
        self, mock_mlflow, mock_avail, base_cfg, tmp_path
    ):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-abc"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with mlflow_context(base_cfg):
            pass

        mock_mlflow.set_tracking_uri.assert_called_once_with(
            (tmp_path / "runs" / "mlflow").as_uri()
        )
        mock_mlflow.set_experiment.assert_called_once_with("checkout_v1")

    @patch("training.mlflow_utils._is_mlflow_available", return_value=True)
    @patch("training.mlflow_utils.mlflow")
    def test_logs_hydra_config(self, mock_mlflow, mock_avail, base_cfg):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-log"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with mlflow_context(base_cfg):
            pass

        # log_params should be called (from _log_hydra_config)
        mock_mlflow.log_params.assert_called_once()
        logged = mock_mlflow.log_params.call_args[0][0]
        assert "model.variant" in logged
        assert "seed" in logged

    @patch("training.mlflow_utils._is_mlflow_available", return_value=True)
    @patch("training.mlflow_utils.mlflow")
    def test_explicit_run_name_passed(self, mock_mlflow, mock_avail, base_cfg):
        cfg = OmegaConf.merge(base_cfg, {"mlflow": {"run_name": "my-run"}})
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-named"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with mlflow_context(cfg):
            pass

        mock_mlflow.start_run.assert_called_once_with(run_name="my-run")

    @patch("training.mlflow_utils._is_mlflow_available", return_value=True)
    @patch("training.mlflow_utils.mlflow")
    def test_restores_env_on_exception(self, mock_mlflow, mock_avail, base_cfg):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-err"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Ensure clean state
        for key in _ENV_KEYS:
            os.environ.pop(key, None)

        with pytest.raises(RuntimeError):
            with mlflow_context(base_cfg):
                raise RuntimeError("training failed")

        # Env vars should still be cleaned up
        for key in _ENV_KEYS:
            assert key not in os.environ, f"{key} was not cleaned up after exception"


# --- Integration with train scripts ---


class TestExcludedKeys:
    def test_mlflow_excluded_from_detector_args(self):
        from training.train_detector import _EXCLUDED_KEYS as detect_excluded

        assert "mlflow" in detect_excluded

    def test_mlflow_excluded_from_classifier_args(self):
        from training.train_classifier import _EXCLUDED_KEYS as classify_excluded

        assert "mlflow" in classify_excluded
