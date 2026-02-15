"""Train YOLO detection model on composite scene dataset."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from src.training.mlflow_utils import mlflow_context

# Keys that are not YOLO training args (belong to other config groups or meta)
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


def build_train_args(
    cfg: DictConfig | dict,
    data_yaml: str | Path,
    device: int = 0,
) -> dict:
    """Build training arguments dict from Hydra config.

    Merges model and training configs, excluding non-training keys.
    The model name is handled separately when constructing the YOLO object.
    """
    training = (
        OmegaConf.to_container(cfg.training, resolve=True)
        if isinstance(cfg, DictConfig) and "training" in cfg
        else {}
    )
    model_cfg = (
        OmegaConf.to_container(cfg.model, resolve=True)
        if isinstance(cfg, DictConfig) and "model" in cfg
        else {}
    )

    args = {}
    # Add training params
    for k, v in training.items():
        if k not in _EXCLUDED_KEYS:
            args[k] = v

    # Add imgsz from model config
    if "imgsz" in model_cfg:
        args["imgsz"] = model_cfg["imgsz"]

    args["data"] = str(data_yaml)
    args["device"] = device
    return args


def get_model_name(cfg: DictConfig | dict) -> str:
    """Extract model name from config."""
    if isinstance(cfg, DictConfig):
        return cfg.model.name
    return cfg.get("model", {}).get("name", "yolo11m.pt")


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


def train(cfg: DictConfig, data_yaml: Path):
    """Run detection training from Hydra config.

    Args:
        cfg: Full Hydra config (with model, training, data sections).
        data_yaml: Path to YOLO data.yaml.
    """
    if cfg.training.get("resume", False):
        checkpoint = _resolve_last_checkpoint(cfg)
        model = YOLO(str(checkpoint))
    else:
        model_name = get_model_name(cfg)
        model = YOLO(model_name)

    device = cfg.get("device", 0)
    train_args = build_train_args(cfg, data_yaml, device=device)

    # Make project path absolute (Hydra changes CWD)
    project_root = Path(cfg.project_root)
    if "project" in train_args:
        train_args["project"] = str(project_root / train_args["project"])

    with mlflow_context(cfg):
        results = model.train(**train_args)
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(cfg.project_root)
    data_yaml = project_root / cfg.data.dataset_dir / "data.yaml"
    print(f"Model: {cfg.model.name}")
    print(f"Data:  {data_yaml}")
    if cfg.training.get("resume", False):
        print("Resuming from last checkpoint")
    print(OmegaConf.to_yaml(cfg.training))
    train(cfg, data_yaml)


if __name__ == "__main__":
    main()
