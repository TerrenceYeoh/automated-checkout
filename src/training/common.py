"""Shared utilities for YOLO training scripts."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Keys that are not YOLO training args (belong to other config groups or meta)
EXCLUDED_KEYS = {
    "model",
    "data",
    "inference",
    "device",
    "project_root",
    "variant",
    "imgsz",
    "mlflow",
}


def resolve_last_checkpoint(cfg: DictConfig) -> Path:
    """Resolve path to last.pt checkpoint for resuming training.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    project_root = Path(cfg.project_root)
    checkpoint = (
        project_root / cfg.training.project / cfg.training.name / "weights" / "last.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def build_train_args(
    cfg: DictConfig | dict,
    data_path: str | Path,
    device: int = 0,
) -> dict:
    """Build YOLO training arguments dict from Hydra config.

    Merges model and training configs, excluding non-training keys.
    Works for both detection and classification training.

    Args:
        cfg: Full Hydra config or plain dict.
        data_path: Path to data.yaml (detection) or data directory (classification).
        device: CUDA device index.

    Returns:
        Dict of keyword arguments for YOLO model.train().
    """
    if isinstance(cfg, DictConfig) and "training" in cfg:
        training = OmegaConf.to_container(cfg.training, resolve=True)
    elif isinstance(cfg, DictConfig):
        training = {}
    else:
        training = dict(cfg)

    args = {}
    for k, v in training.items():
        if k not in EXCLUDED_KEYS:
            args[k] = v

    # Add imgsz from model config
    if isinstance(cfg, DictConfig) and "model" in cfg:
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if "imgsz" in model_cfg:
            args["imgsz"] = model_cfg["imgsz"]

    args["data"] = str(data_path)
    args["device"] = device
    return args
