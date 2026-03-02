"""Tests for background frame extraction from videos."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.extract_backgrounds import extract_all_backgrounds, extract_frames


@pytest.fixture
def mock_video_capture():
    """Mock cv2.VideoCapture that returns 3 frames then stops."""
    with patch("utils.extract_backgrounds.cv2") as mock_cv2:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 90,
        }.get(prop, 0)

        # Return 3 frames then stop
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap.read.side_effect = [
            (True, frame.copy()),
            (True, frame.copy()),
            (True, frame.copy()),
            (False, None),
        ]

        mock_cv2.VideoCapture.return_value = cap
        yield mock_cv2, cap


class TestExtractFrames:
    def test_returns_list_of_paths(self, mock_video_capture, tmp_path):
        mock_cv2, _ = mock_video_capture
        paths = extract_frames(
            Path("video.mp4"), tmp_path, interval_seconds=1.0, max_frames=10
        )
        assert isinstance(paths, list)

    def test_saves_frames_as_jpg(self, mock_video_capture, tmp_path):
        mock_cv2, _ = mock_video_capture
        paths = extract_frames(
            Path("video.mp4"), tmp_path, interval_seconds=1.0, max_frames=10
        )
        assert len(paths) == 3
        for p in paths:
            assert str(p).endswith(".jpg")

    def test_respects_max_frames(self, mock_video_capture, tmp_path):
        mock_cv2, cap = mock_video_capture
        # Reset side_effect to return many frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, frame.copy())] * 10 + [(False, None)]
        paths = extract_frames(
            Path("video.mp4"), tmp_path, interval_seconds=1.0, max_frames=2
        )
        assert len(paths) == 2

    def test_raises_on_unopenable_video(self, tmp_path):
        with patch("utils.extract_backgrounds.cv2") as mock_cv2:
            cap = MagicMock()
            cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = cap
            with pytest.raises(RuntimeError, match="Cannot open video"):
                extract_frames(Path("bad.mp4"), tmp_path)

    def test_creates_output_dir(self, mock_video_capture, tmp_path):
        mock_cv2, _ = mock_video_capture
        out_dir = tmp_path / "sub" / "dir"
        extract_frames(Path("video.mp4"), out_dir, interval_seconds=1.0, max_frames=1)
        assert out_dir.exists()


class TestExtractAllBackgrounds:
    def test_processes_all_mp4_files(self, tmp_path):
        """Should call extract_frames for each .mp4 in the directory."""
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "a.mp4").touch()
        (video_dir / "b.mp4").touch()
        (video_dir / "c.txt").touch()  # should be ignored

        output_dir = tmp_path / "output"

        with patch("utils.extract_backgrounds.extract_frames") as mock_extract:
            mock_extract.return_value = [Path("frame.jpg")]
            paths = extract_all_backgrounds(video_dir, output_dir)
            assert mock_extract.call_count == 2
            assert len(paths) == 2

    def test_empty_directory(self, tmp_path):
        video_dir = tmp_path / "empty"
        video_dir.mkdir()
        output_dir = tmp_path / "output"
        paths = extract_all_backgrounds(video_dir, output_dir)
        assert paths == []
