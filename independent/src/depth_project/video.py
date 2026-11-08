"""Video inspection and ordered RGB decoding."""
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .records import Frame, VideoInfo, sha256_file


def _open(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open/decode video: {path}")
    return capture


def iter_rgb_frames(path: Path, limit: int | None = None) -> Iterator[Frame]:
    if limit is not None and limit < 0:
        raise ValueError("Frame limit must be nonnegative or None")
    capture = _open(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    previous_time = -1.0
    index = 0
    try:
        while limit is None or index < limit:
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not np.isfinite(timestamp) or (index > 0 and timestamp <= previous_time):
                if fps <= 0 or not np.isfinite(fps):
                    raise ValueError("No usable frame timestamps or FPS")
                timestamp = index / fps
            yield Frame(index, float(timestamp), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            previous_time = timestamp
            index += 1
    finally:
        capture.release()


def inspect_video(path: Path, required_frames: int = 100) -> VideoInfo:
    if required_frames < 1:
        raise ValueError("Required frame count must be positive")
    capture = _open(path)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        metadata_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    timestamps = [frame.timestamp_s for frame in iter_rgb_frames(path)]
    if not timestamps:
        raise ValueError(f"No decodable frames: {path}")
    count = len(timestamps)
    return VideoInfo(
        str(Path(path).resolve()), sha256_file(path), width, height, fps,
        metadata_frames, count, required_frames, min(count, required_frames),
        count >= required_frames, timestamps,
    )
