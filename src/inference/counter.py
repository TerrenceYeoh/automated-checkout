"""Product counting from object tracks."""

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TrackRecord:
    """Record of a single tracked object across frames."""

    track_id: int
    detections: list[tuple[int, float]] = field(default_factory=list)
    # Each detection is (class_id, confidence)
    entry_position: tuple[float, float] | None = None
    # Normalized (cx, cy) of the first detection's bounding box center
    first_frame: int | None = None
    last_frame: int | None = None
    last_position: tuple[float, float] | None = None

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


def stitch_tracks(
    tracks: dict[int, TrackRecord],
    max_gap: int = 90,
    max_distance: float = 0.15,
) -> dict[int, TrackRecord]:
    """Merge track fragments likely belonging to the same physical object.

    When the tracker loses an object (e.g. cashier occlusion) and re-acquires it
    with a new ID, the two fragments can be stitched if they are close in time
    and space, and have compatible classes.

    Args:
        tracks: Dict of track_id -> TrackRecord with first_frame/last_frame/last_position set.
        max_gap: Maximum frame gap between end of A and start of B to consider merging.
        max_distance: Maximum normalized Euclidean distance between A's last_position
            and B's entry_position.

    Returns:
        New dict with merged tracks. Absorbed tracks are removed.
    """
    if not tracks:
        return {}

    # Sort track IDs by first_frame so we process earlier tracks first
    sorted_ids = sorted(
        tracks.keys(),
        key=lambda tid: (
            tracks[tid].first_frame if tracks[tid].first_frame is not None else 0
        ),
    )

    # Map each track to its "root" (the track it was merged into)
    root: dict[int, int] = {tid: tid for tid in sorted_ids}

    def find_root(tid: int) -> int:
        while root[tid] != tid:
            root[tid] = root[root[tid]]
            tid = root[tid]
        return tid

    for i, tid_b in enumerate(sorted_ids):
        track_b = tracks[tid_b]
        if track_b.first_frame is None:
            continue

        best_a: int | None = None
        best_gap = max_gap + 1

        for tid_a in sorted_ids[:i]:
            root_a = find_root(tid_a)
            if root_a == tid_b:
                continue
            track_a = tracks[root_a]

            if track_a.last_frame is None:
                continue

            gap = track_b.first_frame - track_a.last_frame
            if gap <= 0 or gap > max_gap:
                continue

            # Distance check
            if track_a.last_position is not None and track_b.entry_position is not None:
                dx = track_a.last_position[0] - track_b.entry_position[0]
                dy = track_a.last_position[1] - track_b.entry_position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > max_distance:
                    continue

            # Class compatibility: skip check if either track is short (<5 detections)
            if track_a.length >= 5 and track_b.length >= 5:
                class_a = majority_vote_class(track_a.detections)
                class_b = majority_vote_class(track_b.detections)
                if class_a != class_b:
                    continue

            if gap < best_gap:
                best_gap = gap
                best_a = root_a

        if best_a is not None:
            # Merge B into A
            track_a = tracks[best_a]
            track_a.detections.extend(track_b.detections)
            if track_b.last_frame is not None:
                track_a.last_frame = track_b.last_frame
            if track_b.last_position is not None:
                track_a.last_position = track_b.last_position
            root[tid_b] = best_a

    # Build result: only include root tracks
    result: dict[int, TrackRecord] = {}
    for tid in sorted_ids:
        if find_root(tid) == tid:
            result[tid] = tracks[tid]
    return result


def filter_by_entry_zone(
    tracks: dict[int, TrackRecord],
    edge: str = "left",
    zone_size: float = 0.15,
) -> dict[int, TrackRecord]:
    """Keep only tracks whose first detection appeared near the specified frame edge.

    This filters out tracks that were re-created mid-frame after an occlusion,
    since genuine new products enter from a known edge of the conveyor.

    Args:
        tracks: Dict of track_id -> TrackRecord (must have entry_position set).
        edge: Which frame edge products enter from: "left", "right", "top", "bottom".
        zone_size: Width of the entry strip as a fraction of frame dimension (0-1).

    Returns:
        Filtered dict containing only tracks within the entry zone.

    Raises:
        ValueError: If edge is not one of "left", "right", "top", "bottom".
    """
    valid_edges = {"left", "right", "top", "bottom"}
    if edge not in valid_edges:
        raise ValueError(f"Invalid edge '{edge}', must be one of {valid_edges}")

    filtered = {}
    for tid, track in tracks.items():
        if track.entry_position is None:
            continue
        cx, cy = track.entry_position
        if edge == "left" and cx <= zone_size:
            filtered[tid] = track
        elif edge == "right" and cx >= 1.0 - zone_size:
            filtered[tid] = track
        elif edge == "top" and cy <= zone_size:
            filtered[tid] = track
        elif edge == "bottom" and cy >= 1.0 - zone_size:
            filtered[tid] = track

    return filtered


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
