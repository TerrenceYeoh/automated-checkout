"""Visualization utilities for debugging scenes, annotations, and detections."""

from pathlib import Path

import cv2
import numpy as np


def draw_yolo_boxes(
    image: np.ndarray,
    label_path: Path,
    class_names: dict[int, str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw YOLO-format bounding boxes on an image.

    Args:
        image: HxWx3 BGR image.
        label_path: Path to YOLO label .txt file.
        class_names: Optional class_id -> name mapping.
        color: Box color in BGR.
        thickness: Line thickness.

    Returns:
        Image with boxes drawn.
    """
    img = image.copy()
    h, w = img.shape[:2]

    if not Path(label_path).exists():
        return img

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h

            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = (
                class_names.get(class_id, str(class_id))
                if class_names
                else str(class_id)
            )
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    return img
