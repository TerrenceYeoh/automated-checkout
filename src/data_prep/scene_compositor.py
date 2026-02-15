"""Synthesize multi-object scenes for YOLO detection training.

Composites RGBA product crops onto background images to create
training scenes that mimic the conveyor belt test environment.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_crop_index(crops_dir: Path) -> dict[int, list[Path]]:
    """Build an index of class_id -> list of crop file paths.

    Expects directory structure: crops_dir/<class_id>/*.png
    """
    crops_dir = Path(crops_dir)
    index: dict[int, list[Path]] = {}
    for class_dir in sorted(crops_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        paths = sorted(class_dir.glob("*.png"))
        if paths:
            index[class_id] = paths
    return index


def paste_product(
    scene: np.ndarray,
    crop: np.ndarray,
    x: int,
    y: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paste an RGBA crop onto a scene at position (x, y) with alpha blending.

    Args:
        scene: HxWx3 RGB background (modified in-place).
        crop: hxwx4 RGBA product crop.
        x: Left edge of placement.
        y: Top edge of placement.

    Returns:
        (modified scene, (x_min, y_min, x_max, y_max) bounding box).
    """
    scene_h, scene_w = scene.shape[:2]
    crop_h, crop_w = crop.shape[:2]

    # Clip to scene bounds
    x_min = max(x, 0)
    y_min = max(y, 0)
    x_max = min(x + crop_w, scene_w)
    y_max = min(y + crop_h, scene_h)

    # Compute corresponding region in the crop
    crop_x_start = x_min - x
    crop_y_start = y_min - y
    crop_x_end = crop_x_start + (x_max - x_min)
    crop_y_end = crop_y_start + (y_max - y_min)

    if x_max <= x_min or y_max <= y_min:
        return scene, (x_min, y_min, x_max, y_max)

    # Extract regions
    crop_region = crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    rgb = crop_region[:, :, :3]
    alpha = crop_region[:, :, 3:4].astype(np.float32) / 255.0

    # Alpha blend
    scene_region = scene[y_min:y_max, x_min:x_max].astype(np.float32)
    blended = rgb.astype(np.float32) * alpha + scene_region * (1.0 - alpha)
    scene[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)

    return scene, (x_min, y_min, x_max, y_max)


def compute_yolo_annotation(
    bbox: tuple[int, int, int, int],
    class_id: int,
    img_w: int,
    img_h: int,
) -> str:
    """Convert a pixel bounding box to YOLO normalized format.

    Args:
        bbox: (x_min, y_min, x_max, y_max) in pixels.
        class_id: YOLO 0-indexed class ID.
        img_w: Image width.
        img_h: Image height.

    Returns:
        YOLO annotation string: "class_id x_center y_center width height"
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def validate_yolo_label(label: str) -> bool:
    """Validate a single YOLO annotation line."""
    parts = label.strip().split()
    if len(parts) != 5:
        return False

    try:
        class_id = int(parts[0])
        if class_id < 0:
            return False
        for val_str in parts[1:]:
            val = float(val_str)
            if val < 0.0 or val > 1.0:
                return False
            # Width and height must be > 0
        width = float(parts[3])
        height = float(parts[4])
        if width <= 0 or height <= 0:
            return False
    except ValueError:
        return False

    return True


def _augment_crop(crop: np.ndarray, target_h: int) -> np.ndarray:
    """Apply random augmentations to a product crop.

    Resizes to fit within the scene and applies transforms.
    """
    h, w = crop.shape[:2]

    # Random scale relative to target scene height
    scale = random.uniform(0.08, 0.35) * target_h / h
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Random rotation
    angle = random.uniform(-30, 30)
    if abs(angle) > 2:
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w - new_w) / 2
        M[1, 2] += (rot_h - new_h) / 2
        crop = cv2.warpAffine(
            crop,
            M,
            (rot_w, rot_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    # Random brightness/contrast on RGB channels only
    if random.random() < 0.5:
        rgb = crop[:, :, :3].astype(np.float32)
        brightness = random.uniform(-30, 30)
        contrast = random.uniform(0.7, 1.3)
        rgb = np.clip(rgb * contrast + brightness, 0, 255).astype(np.uint8)
        crop = np.concatenate([rgb, crop[:, :, 3:]], axis=-1)

    return crop


def compose_scene(
    background: np.ndarray,
    crop_index: dict[int, list[Path]],
    num_products: int,
    class_id_to_yolo: dict[int, int],
    augment: bool = True,
    min_visible_ratio: float = 0.3,
) -> tuple[np.ndarray, list[str]]:
    """Compose a single synthetic scene with multiple products.

    Args:
        background: HxWx3 background image (will be copied).
        crop_index: class_id -> list of crop paths.
        num_products: Number of products to place.
        class_id_to_yolo: original class_id -> YOLO 0-indexed class_id.
        augment: Whether to apply augmentations to crops.
        min_visible_ratio: Minimum visible area ratio for a product to be annotated.

    Returns:
        (scene image HxWx3, list of YOLO annotation strings).
    """
    scene = background.copy()
    scene_h, scene_w = scene.shape[:2]
    annotations = []
    available_classes = list(crop_index.keys())

    for _ in range(num_products):
        # Pick a random class and crop
        class_id = random.choice(available_classes)
        crop_path = random.choice(crop_index[class_id])
        crop = np.array(Image.open(crop_path))

        if crop.ndim != 3 or crop.shape[2] != 4:
            continue

        # Augment
        if augment:
            crop = _augment_crop(crop, scene_h)

        crop_h, crop_w = crop.shape[:2]

        # Random placement position
        x = random.randint(-crop_w // 4, scene_w - crop_w // 2)
        y = random.randint(-crop_h // 4, scene_h - crop_h // 2)

        # Paste onto scene
        scene, bbox = paste_product(scene, crop, x, y)

        x_min, y_min, x_max, y_max = bbox
        visible_w = x_max - x_min
        visible_h = y_max - y_min
        if visible_w <= 0 or visible_h <= 0:
            continue

        # Check minimum visible area
        visible_area = visible_w * visible_h
        full_area = crop_h * crop_w
        if full_area > 0 and (visible_area / full_area) < min_visible_ratio:
            continue

        # Generate YOLO annotation
        yolo_class = class_id_to_yolo.get(class_id)
        if yolo_class is None:
            continue

        ann = compute_yolo_annotation(bbox, yolo_class, scene_w, scene_h)
        if validate_yolo_label(ann):
            annotations.append(ann)

    return scene, annotations


def _apply_scene_augmentations(scene: np.ndarray) -> np.ndarray:
    """Apply global augmentations to the composed scene."""
    # Motion blur (simulates conveyor belt movement)
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        scene = cv2.filter2D(scene, -1, kernel)

    # Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 10), scene.shape).astype(
            np.float32
        )
        scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Color jitter
    if random.random() < 0.4:
        hsv = cv2.cvtColor(scene, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
        scene = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return scene


def generate_dataset(
    crops_dir: Path,
    backgrounds_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    class_id_to_yolo: dict[int, int],
    num_scenes: int = 30000,
    min_products: int = 1,
    max_products: int = 8,
    scene_size: tuple[int, int] = (640, 640),
) -> int:
    """Generate a full synthetic scene dataset.

    Args:
        crops_dir: Directory with class subfolders of RGBA crops.
        backgrounds_dir: Directory with background images.
        output_images_dir: Where to save scene images.
        output_labels_dir: Where to save YOLO label files.
        class_id_to_yolo: Original class ID -> YOLO class ID mapping.
        num_scenes: Number of scenes to generate.
        min_products: Minimum products per scene.
        max_products: Maximum products per scene.
        scene_size: (width, height) of output scenes.

    Returns:
        Number of scenes successfully generated.
    """
    crops_dir = Path(crops_dir)
    backgrounds_dir = Path(backgrounds_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Load crop index
    crop_index = load_crop_index(crops_dir)
    if not crop_index:
        raise ValueError(f"No crops found in {crops_dir}")
    print(
        f"Loaded {sum(len(v) for v in crop_index.values())} crops across {len(crop_index)} classes"
    )

    # Load backgrounds
    bg_paths = list(backgrounds_dir.glob("*.jpg")) + list(backgrounds_dir.glob("*.png"))
    if not bg_paths:
        print("No background images found — using solid gray backgrounds")
        bg_paths = None

    generated = 0
    w, h = scene_size

    for i in tqdm(range(num_scenes), desc="Generating scenes"):
        # Load and resize background
        if bg_paths:
            bg_path = random.choice(bg_paths)
            bg = np.array(Image.open(bg_path).convert("RGB"))
            bg = cv2.resize(bg, (w, h))
        else:
            # Solid gray with slight variation
            gray = random.randint(150, 200)
            bg = np.full((h, w, 3), gray, dtype=np.uint8)

        num_products = random.randint(min_products, max_products)
        scene, annotations = compose_scene(
            bg, crop_index, num_products, class_id_to_yolo
        )

        # Apply global augmentations
        scene = _apply_scene_augmentations(scene)

        # Skip scenes with no valid annotations
        if not annotations:
            continue

        # Save image and labels
        scene_name = f"scene_{i:06d}"
        img_path = output_images_dir / f"{scene_name}.jpg"
        label_path = output_labels_dir / f"{scene_name}.txt"

        Image.fromarray(scene).save(img_path, quality=random.randint(75, 95))
        with open(label_path, "w") as f:
            f.write("\n".join(annotations) + "\n")

        generated += 1

    print(f"Generated {generated} scenes with annotations")
    return generated


if __name__ == "__main__":
    from data_prep.parse_labels import (
        get_original_to_yolo_mapping,
        parse_label_file,
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    labels = parse_label_file(project_root / "data" / "label.txt")
    class_id_to_yolo = get_original_to_yolo_mapping(labels)

    generate_dataset(
        crops_dir=project_root / "data" / "crops",
        backgrounds_dir=project_root / "data" / "backgrounds",
        output_images_dir=project_root / "datasets" / "checkout" / "images" / "all",
        output_labels_dir=project_root / "datasets" / "checkout" / "labels" / "all",
        class_id_to_yolo=class_id_to_yolo,
        num_scenes=30000,
    )
