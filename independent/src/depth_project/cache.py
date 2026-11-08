"""Validated prediction caches with explicit provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def cache_key(source_sha256: str, inference_config: dict) -> str:
    payload = json.dumps({"schema": 1, "source_sha256": source_sha256, "inference": inference_config}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate(values, metadata):
    if values.ndim != 3 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Prediction cache requires finite nonempty THW values")
    frame_ids = metadata.get("frame_ids", [])
    if len(frame_ids) != len(values) or any(a >= b for a,b in zip(frame_ids,frame_ids[1:])):
        raise ValueError("Prediction cache frame IDs/count are inconsistent")


def save_predictions(path: Path, values: np.ndarray, metadata: dict) -> None:
    path = Path(path)
    values = np.asarray(values, dtype=np.float32)
    _validate(values, metadata)
    metadata = {**metadata, "schema": 1, "shape": list(values.shape), "values_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, depths=values, metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_predictions(path: Path, expected_key: str | None = None, expected_frame_ids: list[int] | None = None):
    try:
        with np.load(path, allow_pickle=False) as stored:
            values = stored["depths"].astype(np.float32)
            metadata = json.loads(str(stored["metadata"]))
        _validate(values, metadata)
        if metadata.get("schema") != 1 or metadata.get("shape") != list(values.shape):
            raise ValueError("Prediction cache schema/shape mismatch")
        if metadata.get("values_sha256") != hashlib.sha256(values.tobytes()).hexdigest():
            raise ValueError("Prediction cache checksum mismatch")
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise ValueError(f"Invalid prediction cache {path}: {error}") from error
    if expected_key is not None and metadata.get("key") != expected_key:
        raise ValueError("Prediction cache key mismatch")
    if expected_frame_ids is not None and metadata["frame_ids"] != expected_frame_ids:
        raise ValueError("Prediction cache frame IDs mismatch")
    return values, metadata
