"""Detection evaluation: wrap YOLO model.val() and format results."""

from pathlib import Path

from ultralytics import YOLO


def run_detection_eval(
    model_path: str | Path,
    data_yaml: str | Path,
    eval_cfg: dict,
    model_cfg: dict | None = None,
) -> object:
    """Run YOLO validation and return raw results.

    Args:
        model_path: Path to trained YOLO weights (.pt).
        data_yaml: Path to YOLO data.yaml for the dataset.
        eval_cfg: Evaluation config dict (split, conf_thresh, iou_thresh).
        model_cfg: Model config dict (imgsz). Falls back to eval_cfg if None.

    Returns:
        YOLO validation results object.
    """
    model = YOLO(str(model_path))
    imgsz = (model_cfg or {}).get("imgsz", eval_cfg.get("imgsz", 640))

    results = model.val(
        data=str(data_yaml),
        split=eval_cfg.get("split", "val"),
        conf=eval_cfg.get("conf_thresh", 0.3),
        iou=eval_cfg.get("iou_thresh", 0.45),
        imgsz=imgsz,
        verbose=False,
    )
    return results


def format_detection_results(
    results: object,
    class_names: dict[int, str],
) -> dict:
    """Extract structured metrics from YOLO validation results.

    Args:
        results: Object returned by model.val(), expected to have a
            ``results_dict`` attribute and ``ap_class_index`` / ``box``
            attributes for per-class AP values.
        class_names: Mapping of class index to class name.

    Returns:
        Dict with ``summary`` (mAP50, mAP50-95, precision, recall) and
        ``per_class`` (list of dicts sorted by AP50 ascending, i.e. weakest
        first).
    """
    results_dict = results.results_dict

    summary = {
        "mAP50": results_dict.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": results_dict.get("metrics/mAP50-95(B)", 0.0),
        "precision": results_dict.get("metrics/precision(B)", 0.0),
        "recall": results_dict.get("metrics/recall(B)", 0.0),
    }

    per_class = []
    ap50_values = results.box.ap50
    class_indices = results.box.ap_class_index

    for i, cls_idx in enumerate(class_indices):
        cls_idx = int(cls_idx)
        per_class.append(
            {
                "class_id": cls_idx,
                "class_name": class_names.get(cls_idx, f"class_{cls_idx}"),
                "AP50": float(ap50_values[i]),
            }
        )

    # Sort weakest first (ascending AP50)
    per_class.sort(key=lambda x: x["AP50"])

    return {
        "summary": summary,
        "per_class": per_class,
    }
