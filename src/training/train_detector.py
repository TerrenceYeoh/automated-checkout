"""Train YOLO detection model on composite scene dataset."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from training.common import (
    EXCLUDED_KEYS,
    build_train_args,
    resolve_last_checkpoint,
    transfer_backbone_weights,
)
from training.mlflow_utils import mlflow_context

# Re-export for backwards compatibility with tests
_EXCLUDED_KEYS = EXCLUDED_KEYS
_resolve_last_checkpoint = resolve_last_checkpoint


def get_model_name(cfg: DictConfig | dict) -> str:
    """Extract model name from config."""
    if isinstance(cfg, DictConfig):
        return cfg.model.name
    return cfg.get("model", {}).get("name", "yolo11m.pt")


def train(cfg: DictConfig, data_yaml: str | Path):
    """Run detection training from Hydra config.

    Args:
        cfg: Full Hydra config (with model, training, data sections).
        data_yaml: Path to YOLO data.yaml.
    """
    is_resume = cfg.training.get("resume", False)

    if is_resume:
        checkpoint = resolve_last_checkpoint(cfg)
        model = YOLO(str(checkpoint))
    else:
        model_name = get_model_name(cfg)
        model_path = Path(cfg.project_root) / "models" / model_name
        model = YOLO(str(model_path))

    # Transfer classifier backbone weights (skip when resuming)
    backbone_weights = cfg.model.get("backbone_weights")
    if backbone_weights and not is_resume:
        backbone_path = Path(backbone_weights)
        n = transfer_backbone_weights(model, backbone_path)
        print(f"  Transferred {n} layers from classifier backbone: {backbone_path}")

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
