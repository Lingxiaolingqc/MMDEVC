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


def run_provenance(root) -> dict:
    """Capture at run start, including source hashes when a worktree is dirty."""
    from datetime import datetime,timezone
    from pathlib import Path
    import platform
    import subprocess

    root=Path(root).resolve()
    head=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],text=True,capture_output=True)
    status=subprocess.run(['git','-C',str(root),'status','--porcelain'],text=True,capture_output=True)
    sources={path.relative_to(root).as_posix():sha256_file(path) for path in sorted((root/'src/depth_project').glob('*.py'))}
    configs={path.relative_to(root).as_posix():sha256_file(path) for path in sorted((root/'configs').glob('*.json'))}
    return {'captured_utc':datetime.now(timezone.utc).isoformat(),'git_sha':head.stdout.strip() if head.returncode==0 else None,'git_dirty':bool(status.stdout.strip()) if status.returncode==0 else None,'source_sha256':sources,'config_sha256':configs,'python_version':platform.python_version(),'platform':platform.platform(),'lockfile_sha256':sha256_file(root/'requirements.lock.txt') if (root/'requirements.lock.txt').is_file() else None,'randomness':'frozen networks in evaluation mode; controlled synthetic studies use NumPy seed 42'}
