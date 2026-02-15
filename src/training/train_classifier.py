"""Optional classification pretraining to strengthen backbone features."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from training.mlflow_utils import mlflow_context

_EXCLUDED_KEYS = {
    "model",
    "data",
    "inference",
    "device",
    "project_root",
    "variant",
    "imgsz",
    "mlflow",
}


def build_classify_args(
    cfg: DictConfig | dict,
    data_path: str,
    device: int = 0,
) -> dict:
    """Build classification training arguments from Hydra config."""
    if isinstance(cfg, DictConfig) and "training" in cfg:
        training = OmegaConf.to_container(cfg.training, resolve=True)
    elif isinstance(cfg, DictConfig):
        training = {}
    else:
        training = dict(cfg)

    args = {}
    for k, v in training.items():
        if k not in _EXCLUDED_KEYS:
            args[k] = v

    # Use classification model's imgsz
    if isinstance(cfg, DictConfig) and "model" in cfg:
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if "imgsz" in model_cfg:
            args["imgsz"] = model_cfg["imgsz"]

    args["data"] = data_path
    args["device"] = device
    return args


def _resolve_last_checkpoint(cfg: DictConfig) -> Path:
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


def train_classifier(cfg: DictConfig, data_path: Path):
    """Run classification pretraining from Hydra config."""
    if isinstance(cfg, DictConfig) and cfg.training.get("resume", False):
        checkpoint = _resolve_last_checkpoint(cfg)
        model = YOLO(str(checkpoint))
    else:
        model_name = (
            cfg.model.name
            if isinstance(cfg, DictConfig) and "model" in cfg
            else "yolo11m-cls.pt"
        )
        model = YOLO(model_name)

    device = cfg.get("device", 0) if isinstance(cfg, DictConfig) else 0
    train_args = build_classify_args(cfg, str(data_path), device=device)

    # Make project path absolute (Hydra changes CWD)
    project_root = Path(cfg.project_root)
    if "project" in train_args:
        train_args["project"] = str(project_root / train_args["project"])

    with mlflow_context(cfg):
        results = model.train(**train_args)
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Override to use classification model and training config
    # Run with: python train_classifier.py model=yolo11m_cls training=classify
    project_root = Path(cfg.project_root)
    data_path = project_root / "datasets" / "classification"
    print(f"Model: {cfg.model.name}")
    print(f"Data:  {data_path}")
    train_classifier(cfg, data_path)


if __name__ == "__main__":
    main()
