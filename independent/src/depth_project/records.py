"""Data contracts shared by the independently written pipeline."""
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Frame:
    index: int
    timestamp_s: float
    rgb: np.ndarray


@dataclass(slots=True)
class VideoInfo:
    source: str
    source_sha256: str
    width: int
    height: int
    fps: float
    metadata_frames: int
    decoded_frames: int
    required_frames: int
    available_frames: int
    compliant: bool
    timestamps_s: list[float]


@dataclass(slots=True)
class DepthPrediction:
    values: np.ndarray
    kind: str
    model_id: str
    checkpoint_sha256: str
    preprocessing_id: str
    precision: str

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        if self.values.ndim != 2 or not np.isfinite(self.values).all():
            raise ValueError("Depth must be a finite HW array")
        if self.kind not in {"relative", "metric_m"}:
            raise ValueError("Depth kind must be relative or metric_m")


def sha256_file(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
