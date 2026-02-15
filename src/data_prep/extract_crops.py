"""Extract RGBA crops from training images using segmentation masks."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def threshold_mask(mask: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Threshold a grayscale mask to binary (0 or 255).

    JPEG masks have compression artifacts with intermediate values.
    """
    binary = np.where(mask >= threshold, 255, 0).astype(np.uint8)
    return binary


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Compute bounding box (x_min, y_min, x_max, y_max) from a binary mask.

    Returns None if the mask is empty (all zeros).
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Convert to exclusive upper bounds
    return (int(x_min), int(y_min), int(x_max + 1), int(y_max + 1))


def extract_crop(
    image: np.ndarray, mask: np.ndarray, threshold: int = 128
) -> np.ndarray | None:
    """Extract an RGBA crop of the object from the image using the mask.

    Returns:
        RGBA numpy array cropped to the object bounding box, or None if mask is empty.
    """
    binary_mask = threshold_mask(mask, threshold)
    bbox = bbox_from_mask(binary_mask)
    if bbox is None:
        return None

    x_min, y_min, x_max, y_max = bbox

    # Crop image and mask to bbox
    img_crop = image[y_min:y_max, x_min:x_max]
    mask_crop = binary_mask[y_min:y_max, x_min:x_max]

    # Create RGBA: RGB from image + alpha from mask
    if img_crop.ndim == 2:
        # Grayscale image — convert to 3-channel
        img_crop = np.stack([img_crop] * 3, axis=-1)

    alpha = mask_crop[:, :, np.newaxis]
    rgba = np.concatenate([img_crop, alpha], axis=-1)
    return rgba


def get_image_mask_pairs(
    train_dir: Path, seg_dir: Path
) -> list[tuple[Path, Path, int]]:
    """Find all matching image-mask pairs and extract class IDs.

    Returns:
        List of (image_path, mask_path, class_id) tuples.
    """
    train_dir = Path(train_dir)
    seg_dir = Path(seg_dir)
    pairs = []

    for img_path in sorted(train_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g., "00001_1"
        mask_name = f"{stem}_seg.jpg"
        mask_path = seg_dir / mask_name

        if mask_path.exists():
            # Extract class ID from filename: "00001_1" -> 1
            class_id = int(stem.split("_")[0])
            pairs.append((img_path, mask_path, class_id))

    return pairs


def process_single_image(
    pair: tuple[Path, Path, int], output_dir: Path
) -> tuple[Path, int] | None:
    """Process a single image-mask pair and save the RGBA crop.

    Returns:
        (output_path, class_id) or None if the mask was empty.
    """
    img_path, mask_path, class_id = pair
    output_dir = Path(output_dir)

    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    crop = extract_crop(image, mask)
    if crop is None:
        return None

    # Save to output_dir/<zero-padded class_id>/<original_stem>.png
    class_folder = output_dir / f"{class_id:05d}"
    class_folder.mkdir(parents=True, exist_ok=True)

    crop_path = class_folder / f"{img_path.stem}.png"
    Image.fromarray(crop, mode="RGBA").save(crop_path)

    return (crop_path, class_id)


def extract_all_crops(
    train_dir: Path,
    seg_dir: Path,
    output_dir: Path,
    max_workers: int = 4,
) -> list[tuple[Path, int]]:
    """Extract RGBA crops for all training images.

    Returns:
        List of (output_path, class_id) for successfully processed images.
    """
    pairs = get_image_mask_pairs(train_dir, seg_dir)
    print(f"Found {len(pairs)} image-mask pairs")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, pair, output_dir): pair
            for pair in pairs
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting crops"
        ):
            try:
                result = future.result()
            except Exception as e:
                pair = futures[future]
                print(f"Warning: failed to process {pair[0].name}: {e}")
                continue
            if result is not None:
                results.append(result)

    print(f"Successfully extracted {len(results)} crops")
    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    extract_all_crops(
        train_dir=project_root / "data" / "train",
        seg_dir=project_root / "data" / "segmentation_labels",
        output_dir=project_root / "data" / "crops",
    )
