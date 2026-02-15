"""Product counting from object tracks."""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TrackRecord:
    """Record of a single tracked object across frames."""

    track_id: int
    detections: list[tuple[int, float]] = field(default_factory=list)
    # Each detection is (class_id, confidence)

    def add_detection(self, class_id: int, confidence: float):
        self.detections.append((class_id, confidence))

    @property
    def length(self) -> int:
        return len(self.detections)


def majority_vote_class(detections: list[tuple[int, float]]) -> int:
    """Determine the class of a track via majority voting.

    Ties are broken by total confidence sum.

    Args:
        detections: List of (class_id, confidence) per frame.

    Returns:
        The winning class ID.
    """
    if not detections:
        raise ValueError("Cannot vote on empty detections list")

    # Count occurrences and sum confidences per class
    class_counts: dict[int, int] = Counter()
    class_conf: dict[int, float] = {}
    for class_id, conf in detections:
        class_counts[class_id] += 1
        class_conf[class_id] = class_conf.get(class_id, 0.0) + conf

    # Sort by (count descending, total confidence descending)
    best = max(
        class_counts.keys(),
        key=lambda c: (class_counts[c], class_conf[c]),
    )
    return best


def filter_short_tracks(
    tracks: dict[int, TrackRecord], min_length: int = 5
) -> dict[int, TrackRecord]:
    """Remove tracks shorter than min_length frames."""
    return {tid: track for tid, track in tracks.items() if track.length >= min_length}


def count_products(tracks: dict[int, TrackRecord]) -> dict[int, int]:
    """Count products by class from finalized tracks.

    Uses majority voting to assign each track a single class,
    then counts unique tracks per class.

    Returns:
        Dict mapping class_id to count.
    """
    counts: dict[int, int] = Counter()
    for track in tracks.values():
        if not track.detections:
            continue
        class_id = majority_vote_class(track.detections)
        counts[class_id] += 1
    return dict(counts)
