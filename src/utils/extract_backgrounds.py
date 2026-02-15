"""Extract conveyor belt background frames from test videos.

Extracts frames at regular intervals. These frames serve as backgrounds
for the scene compositor, helping bridge the synthetic-to-real domain gap.
"""

from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 5.0,
    max_frames: int = 50,
) -> list[Path]:
    """Extract frames from a video at regular intervals.

    Args:
        video_path: Path to the MP4 video.
        output_dir: Directory to save extracted frames.
        interval_seconds: Seconds between extracted frames.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of saved frame paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(fps * interval_seconds)

    saved_paths = []
    frame_idx = 0
    extracted = 0

    with tqdm(
        total=min(max_frames, total_frames // max(interval_frames, 1)),
        desc=f"Extracting from {video_path.name}",
    ) as pbar:
        while cap.isOpened() and extracted < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Save as JPEG
            fname = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            out_path = output_dir / fname
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_paths.append(out_path)

            extracted += 1
            frame_idx += interval_frames
            pbar.update(1)

    cap.release()
    return saved_paths


def extract_all_backgrounds(
    video_dir: Path,
    output_dir: Path,
    interval_seconds: float = 5.0,
    max_per_video: int = 50,
) -> list[Path]:
    """Extract background frames from all test videos.

    Args:
        video_dir: Directory containing MP4 test videos.
        output_dir: Where to save extracted frames.
        interval_seconds: Seconds between frames.
        max_per_video: Max frames per video.

    Returns:
        List of all saved frame paths.
    """
    video_dir = Path(video_dir)
    all_paths = []

    for video_path in sorted(video_dir.glob("*.mp4")):
        paths = extract_frames(
            video_path,
            output_dir,
            interval_seconds=interval_seconds,
            max_frames=max_per_video,
        )
        all_paths.extend(paths)
        print(f"  {video_path.name}: {len(paths)} frames")

    print(f"Total background frames extracted: {len(all_paths)}")
    return all_paths


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    extract_all_backgrounds(
        video_dir=project_root / "data" / "testA",
        output_dir=project_root / "data" / "backgrounds",
        interval_seconds=5.0,
        max_per_video=50,
    )
