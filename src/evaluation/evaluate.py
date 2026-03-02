"""Evaluation entry point: detection metrics, counting metrics, or both."""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from data_prep.parse_labels import get_class_mapping, parse_label_file
from evaluation.counting_metrics import evaluate_all_videos
from evaluation.detection_metrics import format_detection_results, run_detection_eval


def _resolve_model_path(cfg: DictConfig) -> Path:
    """Resolve trained model weights path from config."""
    project_root = Path(cfg.project_root)
    return (
        project_root / cfg.training.project / cfg.training.name / "weights" / "best.pt"
    )


def _print_detection_summary(results: dict, n_weakest: int = 10) -> None:
    """Print formatted detection evaluation results."""
    s = results["summary"]
    print("\n=== Detection Evaluation ===")
    print(f"  mAP@50:    {s['mAP50']:.4f}")
    print(f"  mAP@50-95: {s['mAP50-95']:.4f}")
    print(f"  Precision: {s['precision']:.4f}")
    print(f"  Recall:    {s['recall']:.4f}")

    per_class = results["per_class"]
    if per_class:
        weakest = per_class[:n_weakest]
        print(f"\n  Weakest {len(weakest)} classes (by AP@50):")
        for entry in weakest:
            print(f"    {entry['class_name']:30s}  AP50={entry['AP50']:.4f}")

        strongest = per_class[-n_weakest:][::-1]
        print(f"\n  Strongest {len(strongest)} classes (by AP@50):")
        for entry in strongest:
            print(f"    {entry['class_name']:30s}  AP50={entry['AP50']:.4f}")


def _print_counting_summary(results: dict) -> None:
    """Print formatted counting evaluation results."""
    agg = results["aggregate"]
    print("\n=== Counting Evaluation ===")
    print(f"  Videos evaluated: {agg['total_videos']}")
    print(f"  Aggregate MAE:    {agg['mae']:.4f}")
    print(f"  Class accuracy:   {agg['class_accuracy']:.4f}")

    if agg["unmatched_predictions"]:
        print(f"  Unmatched predictions: {agg['unmatched_predictions']}")
    if agg["unmatched_ground_truth"]:
        print(f"  Unmatched ground truth: {agg['unmatched_ground_truth']}")

    for video in results["per_video"]:
        print(f"\n  Video: {video['video']}")
        print(
            f"    MAE: {video['mae']:.4f}  Class accuracy: {video['class_accuracy']:.4f}"
        )
        if video["missed_classes"]:
            print(f"    Missed classes: {video['missed_classes']}")
        if video["spurious_classes"]:
            print(f"    Spurious classes: {video['spurious_classes']}")


def run_evaluation(cfg: DictConfig) -> dict:
    """Run evaluation based on config mode.

    Args:
        cfg: Full Hydra config.

    Returns:
        Dict with evaluation results (detection, counting, or both).
    """
    project_root = Path(cfg.project_root)
    eval_cfg = OmegaConf.to_container(cfg.evaluation, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    mode = eval_cfg.get("mode", "detection")

    results = {}

    if mode in ("detection", "both"):
        model_path = _resolve_model_path(cfg)
        data_yaml = project_root / cfg.data.dataset_dir / "data.yaml"

        labels = parse_label_file(project_root / cfg.data.label_file)
        class_names = get_class_mapping(labels)

        print(f"Model: {model_path}")
        print(f"Data:  {data_yaml}")
        print(f"Split: {eval_cfg['split']}")
        print(f"Conf:  {eval_cfg['conf_thresh']}  IoU: {eval_cfg['iou_thresh']}")

        raw_results = run_detection_eval(model_path, data_yaml, eval_cfg, model_cfg)
        detection_results = format_detection_results(raw_results, class_names)
        results["detection"] = detection_results
        _print_detection_summary(detection_results)

    if mode in ("counting", "both"):
        gt_dir = eval_cfg.get("ground_truth_dir")
        pred_dir = eval_cfg.get("predictions_dir")

        if gt_dir is None or pred_dir is None:
            print(
                "\nSkipping counting evaluation: "
                "ground_truth_dir and predictions_dir must be set."
            )
        else:
            gt_dir = project_root / gt_dir
            pred_dir = project_root / pred_dir
            counting_results = evaluate_all_videos(pred_dir, gt_dir)
            results["counting"] = counting_results
            _print_counting_summary(counting_results)

    # Save results
    if eval_cfg.get("save_results", True):
        output_dir = project_root / eval_cfg.get("output_dir", "outputs/eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "eval_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Evaluation config:\n{OmegaConf.to_yaml(cfg.evaluation)}")
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
