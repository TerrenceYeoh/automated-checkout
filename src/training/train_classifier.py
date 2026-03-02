"""Optional classification pretraining to strengthen backbone features."""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO

from training.common import EXCLUDED_KEYS, build_train_args, resolve_last_checkpoint
from training.mlflow_utils import mlflow_context

# Re-export for backwards compatibility with tests
_EXCLUDED_KEYS = EXCLUDED_KEYS
_resolve_last_checkpoint = resolve_last_checkpoint

# Keep old name as alias so existing test imports still work
build_classify_args = build_train_args


def train_classifier(cfg: DictConfig, data_path: str | Path):
    """Run classification pretraining from Hydra config."""
    if isinstance(cfg, DictConfig) and cfg.training.get("resume", False):
        checkpoint = resolve_last_checkpoint(cfg)
        model = YOLO(str(checkpoint))
    else:
        model_name = (
            cfg.model.name
            if isinstance(cfg, DictConfig) and "model" in cfg
            else "yolo11m-cls.pt"
        )
        model_path = Path(cfg.project_root) / "models" / model_name
        model = YOLO(str(model_path))

    device = cfg.get("device", 0) if isinstance(cfg, DictConfig) else 0
    train_args = build_train_args(cfg, str(data_path), device=device)

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
