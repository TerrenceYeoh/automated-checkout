"""Counting evaluation: compare predicted vs ground truth product counts."""

import json
from pathlib import Path


def load_count_json(path: str | Path) -> dict[int, int]:
    """Load an inference results JSON and extract class_id -> count mapping.

    Expects the JSON format produced by ``run_inference.format_results``:
    ``{"products": [{"class_id": int, "count": int}, ...]}``.

    Args:
        path: Path to a JSON results file.

    Returns:
        Dict mapping class_id to count.
    """
    with open(path) as f:
        data = json.load(f)

    counts: dict[int, int] = {}
    for product in data["products"]:
        counts[int(product["class_id"])] = int(product["count"])
    return counts


def compute_counting_metrics(
    predictions: dict[int, int],
    ground_truth: dict[int, int],
) -> dict:
    """Compute counting metrics for a single video.

    Args:
        predictions: class_id -> predicted count.
        ground_truth: class_id -> true count.

    Returns:
        Dict with per_class errors, MAE, class_accuracy,
        missed_classes, and spurious_classes.
    """
    all_classes = sorted(set(predictions) | set(ground_truth))

    per_class = []
    correct = 0
    total_abs_error = 0

    for cls_id in all_classes:
        pred = predictions.get(cls_id, 0)
        gt = ground_truth.get(cls_id, 0)
        error = pred - gt
        abs_error = abs(error)
        total_abs_error += abs_error
        if pred == gt:
            correct += 1
        per_class.append(
            {
                "class_id": cls_id,
                "predicted": pred,
                "ground_truth": gt,
                "error": error,
                "abs_error": abs_error,
            }
        )

    n_classes = len(all_classes) if all_classes else 1
    mae = total_abs_error / n_classes
    class_accuracy = correct / n_classes

    missed_classes = sorted(set(ground_truth) - set(predictions))
    spurious_classes = sorted(set(predictions) - set(ground_truth))

    return {
        "per_class": per_class,
        "mae": mae,
        "class_accuracy": class_accuracy,
        "missed_classes": missed_classes,
        "spurious_classes": spurious_classes,
        "total_abs_error": total_abs_error,
        "num_classes": len(all_classes),
    }


def evaluate_all_videos(
    predictions_dir: str | Path,
    ground_truth_dir: str | Path,
) -> dict:
    """Match prediction and ground-truth files by stem and aggregate metrics.

    Matches ``<stem>_results.json`` in predictions_dir to
    ``<stem>_gt.json`` in ground_truth_dir by extracting the video stem.

    Args:
        predictions_dir: Directory containing ``*_results.json`` files.
        ground_truth_dir: Directory containing ``*_gt.json`` files.

    Returns:
        Dict with per_video results and aggregated metrics.
    """
    pred_dir = Path(predictions_dir)
    gt_dir = Path(ground_truth_dir)

    # Build mapping: stem -> path
    pred_files = {
        p.stem.removesuffix("_results"): p for p in pred_dir.glob("*_results.json")
    }
    gt_files = {g.stem.removesuffix("_gt"): g for g in gt_dir.glob("*_gt.json")}

    matched_stems = sorted(set(pred_files) & set(gt_files))

    per_video = []
    total_abs_error = 0
    total_classes = 0
    total_correct = 0

    for stem in matched_stems:
        preds = load_count_json(pred_files[stem])
        gt = load_count_json(gt_files[stem])
        metrics = compute_counting_metrics(preds, gt)

        per_video.append(
            {
                "video": stem,
                **metrics,
            }
        )

        total_abs_error += metrics["total_abs_error"]
        total_classes += metrics["num_classes"]
        n_correct = sum(1 for entry in metrics["per_class"] if entry["abs_error"] == 0)
        total_correct += n_correct

    aggregate_mae = total_abs_error / total_classes if total_classes > 0 else 0.0
    aggregate_accuracy = total_correct / total_classes if total_classes > 0 else 0.0

    return {
        "per_video": per_video,
        "aggregate": {
            "mae": aggregate_mae,
            "class_accuracy": aggregate_accuracy,
            "total_videos": len(matched_stems),
            "unmatched_predictions": sorted(set(pred_files) - set(gt_files)),
            "unmatched_ground_truth": sorted(set(gt_files) - set(pred_files)),
        },
    }
