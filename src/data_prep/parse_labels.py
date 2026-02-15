"""Parse label.txt and generate YOLO-format data.yaml."""

import warnings
from pathlib import Path

import yaml


def parse_label_file(label_path: Path) -> dict[int, str]:
    """Parse label.txt into a mapping of original class ID to class name.

    Format per line: "ClassName,ID"

    Returns:
        Dict mapping original 1-indexed class ID (int) to class name (str).
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    labels = {}
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "Name,ID" — split from the right to handle commas in names
            parts = line.rsplit(",", maxsplit=1)
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            try:
                class_id = int(parts[1].strip())
            except ValueError:
                continue
            if class_id in labels:
                warnings.warn(
                    f"Duplicate class ID {class_id}: '{labels[class_id]}' -> '{name}'",
                    stacklevel=2,
                )
            labels[class_id] = name
    return labels


def get_class_mapping(labels: dict[int, str]) -> dict[int, str]:
    """Convert original 1-indexed class IDs to 0-indexed YOLO class IDs.

    Sorts by original ID and remaps to contiguous 0..N-1.

    Returns:
        Dict mapping YOLO class ID (0-indexed) to class name.
    """
    sorted_items = sorted(labels.items(), key=lambda x: x[0])
    return {i: name for i, (_, name) in enumerate(sorted_items)}


def get_original_to_yolo_mapping(labels: dict[int, str]) -> dict[int, int]:
    """Get a mapping from original class IDs to YOLO 0-indexed class IDs.

    Returns:
        Dict mapping original class ID to YOLO class ID.
    """
    sorted_ids = sorted(labels.keys())
    return {orig_id: yolo_id for yolo_id, orig_id in enumerate(sorted_ids)}


def generate_data_yaml(
    class_mapping: dict[int, str],
    output_path: Path,
    dataset_path: str = "./datasets/checkout",
    include_test: bool = False,
) -> None:
    """Generate a YOLO-format data.yaml file.

    Args:
        class_mapping: 0-indexed YOLO class ID to class name mapping.
        output_path: Where to write the YAML file.
        dataset_path: Root path of the dataset (relative or absolute).
        include_test: Whether to include a test split path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
    }
    if include_test:
        data["test"] = "images/test"
    data["nc"] = len(class_mapping)
    data["names"] = class_mapping

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    label_path = project_root / "data" / "label.txt"
    output_dir = project_root / "datasets" / "checkout"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = parse_label_file(label_path)
    print(f"Parsed {len(labels)} classes from {label_path}")

    mapping = get_class_mapping(labels)
    yaml_path = output_dir / "data.yaml"
    generate_data_yaml(mapping, yaml_path)
    print(f"Generated {yaml_path}")
