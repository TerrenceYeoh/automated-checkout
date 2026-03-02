"""Tests for the counting and track post-processing logic."""

import pytest

from inference.counter import (
    TrackRecord,
    count_products,
    filter_short_tracks,
    majority_vote_class,
)


class TestMajorityVoteClass:
    def test_unanimous(self):
        detections = [(5, 0.9), (5, 0.8), (5, 0.95)]
        assert majority_vote_class(detections) == 5

    def test_majority(self):
        detections = [(5, 0.9), (5, 0.8), (3, 0.95), (5, 0.7)]
        assert majority_vote_class(detections) == 5

    def test_tie_breaks_by_confidence(self):
        """When two classes appear equally often, pick the one with higher total confidence."""
        detections = [(5, 0.9), (5, 0.8), (3, 0.95), (3, 0.99)]
        # Class 5: count=2, total_conf=1.7
        # Class 3: count=2, total_conf=1.94
        assert majority_vote_class(detections) == 3

    def test_single_detection(self):
        assert majority_vote_class([(10, 0.5)]) == 10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            majority_vote_class([])


class TestFilterShortTracks:
    def test_filters_by_length(self):
        tracks = {
            1: TrackRecord(track_id=1, detections=[(0, 0.9)] * 10),
            2: TrackRecord(track_id=2, detections=[(0, 0.9)] * 3),
            3: TrackRecord(track_id=3, detections=[(0, 0.9)] * 7),
        }
        filtered = filter_short_tracks(tracks, min_length=5)
        assert set(filtered.keys()) == {1, 3}

    def test_keeps_all_if_above_threshold(self):
        tracks = {
            1: TrackRecord(track_id=1, detections=[(0, 0.9)] * 10),
            2: TrackRecord(track_id=2, detections=[(0, 0.9)] * 10),
        }
        filtered = filter_short_tracks(tracks, min_length=5)
        assert len(filtered) == 2

    def test_removes_all_if_below_threshold(self):
        tracks = {
            1: TrackRecord(track_id=1, detections=[(0, 0.9)] * 2),
            2: TrackRecord(track_id=2, detections=[(0, 0.9)] * 1),
        }
        filtered = filter_short_tracks(tracks, min_length=5)
        assert len(filtered) == 0


class TestCountProducts:
    def test_basic_counting(self):
        tracks = {
            1: TrackRecord(track_id=1, detections=[(0, 0.9)] * 10),
            2: TrackRecord(track_id=2, detections=[(1, 0.8)] * 8),
            3: TrackRecord(track_id=3, detections=[(0, 0.9)] * 6),
        }
        counts = count_products(tracks)
        assert counts[0] == 2  # two tracks of class 0
        assert counts[1] == 1  # one track of class 1

    def test_empty_tracks(self):
        counts = count_products({})
        assert len(counts) == 0

    def test_majority_vote_applied(self):
        """Track with mixed detections should use majority vote."""
        tracks = {
            1: TrackRecord(
                track_id=1,
                detections=[(5, 0.9), (5, 0.8), (5, 0.7), (3, 0.95), (5, 0.85)],
            ),
        }
        counts = count_products(tracks)
        assert counts[5] == 1
        assert counts.get(3, 0) == 0
