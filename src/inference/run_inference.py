"""End-to-end video inference pipeline: detect, track, count."""

import json
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from inference.counter import TrackRecord, count_products, filter_short_tracks


def format_results(
    video_name: str,
    counts: dict[int, int],
    class_names: dict[int, str],
) -> dict:
    """Format counting results into a structured dict.

    Args:
        video_name: Name of the video file.
        counts: class_id -> count mapping.
        class_names: class_id -> class name mapping.

    Returns:
        Structured result dict, JSON-serializable.
    """
    products = []
    for class_id in sorted(counts.keys()):
        products.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"unknown_{class_id}"),
                "count": counts[class_id],
            }
        )

    return {
        "video": video_name,
        "total_products": sum(counts.values()),
        "unique_classes": len(counts),
        "products": products,
    }


def _init_video_writer(video_path: Path, output_path: Path) -> cv2.VideoWriter:
    """Create a VideoWriter matching the source video's FPS and frame size.

    Args:
        video_path: Source video to read properties from.
        output_path: Path for the output annotated video.

    Returns:
        Configured cv2.VideoWriter ready for writing frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def run_video_inference(
    model_path: Path,
    video_path: Path,
    inference_cfg: dict,
    model_cfg: dict | None = None,
    visualize: bool = False,
    show_live: bool = False,
    output_dir: Path | None = None,
) -> dict[int, TrackRecord]:
    """Run detection + tracking on a single video.

    Args:
        model_path: Path to YOLO weights.
        video_path: Path to video file.
        inference_cfg: Inference config (conf_thresh, iou_thresh, tracker, etc.).
        model_cfg: Model config (imgsz). Falls back to inference_cfg if not provided.
        visualize: If True, save annotated video to output_dir.
        show_live: If True, display annotated frames in a window.
        output_dir: Directory for saving annotated video (required if visualize=True).

    Returns:
        Dict of track_id -> TrackRecord with all detections.
    """
    model = YOLO(str(model_path))
    tracks: dict[int, TrackRecord] = {}

    writer = None
    if visualize:
        output_path = Path(output_dir) / f"{video_path.stem}_annotated.mp4"
        writer = _init_video_writer(video_path, output_path)

    imgsz = (model_cfg or {}).get("imgsz", inference_cfg.get("imgsz", 640))

    results_generator = model.track(
        source=str(video_path),
        tracker=inference_cfg.get("tracker", "bytetrack.yaml"),
        conf=inference_cfg.get("conf_thresh", 0.3),
        iou=inference_cfg.get("iou_thresh", 0.45),
        imgsz=imgsz,
        persist=inference_cfg.get("persist", True),
        stream=True,
        verbose=False,
    )

    for result in results_generator:
        if visualize or show_live:
            has_detections = result.boxes is not None and result.boxes.id is not None
            frame = result.plot() if has_detections else result.orig_img

            if writer is not None:
                writer.write(frame)

            if show_live:
                cv2.imshow("Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for tid, cid, conf in zip(track_ids, class_ids, confidences, strict=True):
            if tid not in tracks:
                tracks[tid] = TrackRecord(track_id=tid)
            tracks[tid].add_detection(cid, conf)

    if writer is not None:
        writer.release()
    if show_live:
        cv2.destroyAllWindows()

    return tracks


def run_pipeline(
    cfg: DictConfig,
    model_path: Path,
    class_names: dict[int, str],
) -> None:
    """Run the full inference pipeline on all test videos.

    Args:
        cfg: Full Hydra config.
        model_path: Path to trained YOLO model weights.
        class_names: YOLO class_id -> class name mapping.
    """
    project_root = Path(cfg.project_root)
    video_dir = project_root / cfg.data.video_dir
    output_dir = project_root / cfg.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_cfg = OmegaConf.to_container(cfg.inference, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    video_paths = sorted(video_dir.glob("*.mp4"))
    if not video_paths:
        print(f"No MP4 files found in {video_dir}")
        return

    min_track_len = inference_cfg.get("min_track_length", 5)

    visualize = inference_cfg.get("visualize", False)
    show_live = inference_cfg.get("show_live", False)

    for video_path in video_paths:
        print(f"\nProcessing {video_path.name}...")

        raw_tracks = run_video_inference(
            model_path,
            video_path,
            inference_cfg,
            model_cfg,
            visualize=visualize,
            show_live=show_live,
            output_dir=output_dir,
        )
        print(f"  Raw tracks: {len(raw_tracks)}")

        tracks = filter_short_tracks(raw_tracks, min_length=min_track_len)
        print(f"  After filtering (min {min_track_len} frames): {len(tracks)}")

        counts = count_products(tracks)
        print(
            f"  Products detected: {sum(counts.values())} items across {len(counts)} classes"
        )

        result = format_results(video_path.name, counts, class_names)
        output_file = output_dir / f"{video_path.stem}_results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to {output_file}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from data_prep.parse_labels import get_class_mapping, parse_label_file

    project_root = Path(cfg.project_root)
    labels = parse_label_file(project_root / cfg.data.label_file)
    class_names = get_class_mapping(labels)

    model_path = (
        project_root / cfg.training.project / cfg.training.name / "weights" / "best.pt"
    )
    print(f"Model: {model_path}")
    print(f"Inference config:\n{OmegaConf.to_yaml(cfg.inference)}")

    viz_modes = []
    if cfg.inference.get("visualize", False):
        viz_modes.append("save video")
    if cfg.inference.get("show_live", False):
        viz_modes.append("live display")
    if viz_modes:
        print(f"Visualization: {', '.join(viz_modes)}")

    run_pipeline(cfg, model_path, class_names)


if __name__ == "__main__":
    main()
