"""Shared utilities for YOLO training scripts."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

# Keys that are not YOLO training args (belong to other config groups or meta)
EXCLUDED_KEYS = {
    "model",
    "data",
    "inference",
    "evaluation",
    "device",
    "project_root",
    "variant",
    "imgsz",
    "mlflow",
    "backbone_weights",
}


def transfer_backbone_weights(
    detector: YOLO,
    classifier_weights: str | Path,
) -> int:
    """Transfer matching backbone weights from classifier to detector.

    Uses strict=False to skip mismatched head layers.
    Returns number of transferred parameters.
    """
    cls_model = YOLO(str(classifier_weights))
    cls_state = cls_model.model.state_dict()
    det_state = detector.model.state_dict()

    # Find matching keys (same name + same shape)
    transferred = {}
    for key in det_state:
        if key in cls_state and cls_state[key].shape == det_state[key].shape:
            transferred[key] = cls_state[key]

    detector.model.load_state_dict(transferred, strict=False)
    return len(transferred)


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
