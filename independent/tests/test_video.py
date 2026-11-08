"""Catch incorrect frame counting, frame order, RGB convention, and corrupt input."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from depth_project import video


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "three_frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (32, 24))
    assert writer.isOpened()
    for bgr in [(0, 0, 255), (255, 0, 0), (0, 255, 0)]:
        writer.write(np.full((24, 32, 3), bgr, np.uint8))
    writer.release()
    return path


def test_counts_decoded_frames_not_requested_frames(clip):
    info = video.inspect_video(clip, required_frames=100)
    assert info.decoded_frames == 3
    assert (info.width, info.height) == (32, 24)
    assert info.fps == pytest.approx(12)
    assert info.available_frames == 3
    assert not info.compliant
    assert len(info.source_sha256) == 64


def test_frame_order_and_rgb_conversion(clip):
    frames = list(video.iter_rgb_frames(clip, limit=2))
    assert [f.index for f in frames] == [0, 1]
    assert frames[0].rgb.shape == (24, 32, 3)
    assert frames[0].rgb.dtype == np.uint8
    np.testing.assert_allclose(frames[0].rgb[12, 16], [255, 0, 0], atol=4)
    np.testing.assert_allclose(frames[1].rgb[12, 16], [0, 0, 255], atol=4)
    assert frames[1].timestamp_s == pytest.approx(1 / 12, abs=0.001)


def test_corrupt_video_is_error(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="decode|open"):
        video.inspect_video(path)


def test_missing_video_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        video.inspect_video(tmp_path / "missing.mp4")


def test_invalid_limit_is_error(clip):
    with pytest.raises(ValueError):
        list(video.iter_rgb_frames(clip, limit=-1))
