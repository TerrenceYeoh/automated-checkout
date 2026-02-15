"""MLflow experiment tracking integration for Ultralytics training.

Provides a context manager that configures the environment so that the
Ultralytics built-in MLflow callback logs into the correct experiment.
Also logs supplementary Hydra config data that the callback doesn't capture.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]


def _is_mlflow_available() -> bool:
    """Check if mlflow is installed."""
    return mlflow is not None


def _resolve_tracking_uri(cfg: DictConfig) -> str:
    """Resolve tracking_uri to an absolute file URI.

    If cfg.mlflow.tracking_uri is a relative path, it is resolved
    relative to cfg.project_root. Local paths are converted to file:///
    URIs as required by MLflow 3.x.
    """
    raw_uri = cfg.mlflow.tracking_uri

    # Already has a URI scheme (e.g. http://, databricks://, file://)
    if raw_uri is not None and "://" in raw_uri:
        return raw_uri

    # Resolve to absolute path
    if raw_uri is None:
        abs_path = Path(cfg.project_root) / "runs" / "mlflow"
    elif Path(raw_uri).is_absolute():
        abs_path = Path(raw_uri)
    else:
        abs_path = Path(cfg.project_root) / raw_uri

    # MLflow 3.x requires file:/// URI for local file stores
    return abs_path.as_uri()


def _resolve_experiment_name(cfg: DictConfig) -> str:
    """Derive experiment name from config.

    Uses cfg.mlflow.experiment_name if set, otherwise falls back to
    cfg.training.name (the YOLO run name).
    """
    explicit = cfg.mlflow.get("experiment_name")
    if explicit is not None:
        return explicit

    # Fall back to training run name
    if "training" in cfg and "name" in cfg.training:
        return cfg.training.name

    return "Default"


def _log_hydra_config(cfg: DictConfig) -> None:
    """Log supplementary Hydra config params that Ultralytics doesn't capture."""
    params: dict[str, Any] = {}

    # Model config
    if "model" in cfg:
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        for key in ("variant", "imgsz"):
            if key in model_cfg:
                params[f"model.{key}"] = model_cfg[key]

    # Data config
    if "data" in cfg:
        data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
        for key in ("dataset_dir", "label_file", "num_scenes", "val_ratio"):
            if key in data_cfg:
                params[f"data.{key}"] = data_cfg[key]

    # Global settings
    for key in ("device", "seed"):
        if key in cfg:
            params[key] = cfg[key]

    if params:
        mlflow.log_params(params)


def _start_mlflow_ui(tracking_uri: str, port: int) -> subprocess.Popen | None:
    """Launch MLflow UI as a background process.

    Returns the process handle, or None if it fails to start.
    """
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                tracking_uri,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Brief pause to let the server bind
        time.sleep(1)
        if proc.poll() is not None:
            # Process exited immediately (port in use, etc.)
            return None
        return proc
    except Exception:
        return None


_ENV_KEYS = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_RUN")


@contextmanager
def mlflow_context(cfg: DictConfig) -> Generator[None, None, None]:
    """Set up MLflow environment for Ultralytics built-in callback.

    Usage::

        with mlflow_context(cfg):
            results = model.train(**train_args)

    When mlflow is disabled or not installed, this is a no-op.
    """
    # Check if tracking is enabled
    if "mlflow" not in cfg or not cfg.mlflow.get("enabled", False):
        yield
        return

    # Check if mlflow is installed
    if not _is_mlflow_available():
        yield
        return

    # Save original env vars for cleanup
    saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}

    tracking_uri = _resolve_tracking_uri(cfg)
    experiment_name = _resolve_experiment_name(cfg)

    # Launch MLflow UI
    ui_port = cfg.mlflow.get("ui_port", 5000)
    ui_proc = _start_mlflow_ui(tracking_uri, ui_port)
    if ui_proc is not None:
        print(f"MLflow UI: http://localhost:{ui_port}")
    else:
        print(f"MLflow UI: failed to start on port {ui_port} (may already be running)")

    try:
        # Set env vars for the Ultralytics MLflow callback
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

        run_name = cfg.mlflow.get("run_name")
        if run_name is not None:
            os.environ["MLFLOW_RUN"] = run_name
        elif "MLFLOW_RUN" in os.environ:
            del os.environ["MLFLOW_RUN"]

        # Set tracking URI for the mlflow client
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Pre-start run so Ultralytics callback joins it
        with mlflow.start_run(run_name=run_name) as run:
            # Log supplementary Hydra config
            _log_hydra_config(cfg)
            # Set MLFLOW_RUN to active run ID so Ultralytics callback detects it
            os.environ["MLFLOW_RUN"] = run.info.run_id

            yield
    finally:
        # Restore original env vars
        for key, original in saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

        # Shut down MLflow UI
        if ui_proc is not None and ui_proc.poll() is None:
            ui_proc.terminate()
            ui_proc.wait(timeout=5)
