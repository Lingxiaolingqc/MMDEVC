"""Original NYUv2 labeled MATLAB v7.3 data, without quantizing meter-valued GT."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


NYU_SHAPE = (480, 640)
NYU_CROP_YX = (45, 471, 41, 601)  # UniDepth official NYUv2 eval_mask; Python half-open.


def read_standard_split(path):
    values = loadmat(path)
    result = {}
    for name, key, expected in (("train", "trainNdxs", 795), ("test", "testNdxs", 654)):
        raw = np.asarray(values[key]).ravel()
        if len(raw) != expected or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError(f"Invalid {key}: expected {expected} integer indices")
        indices = raw.astype(np.int64)
        if len(np.unique(indices)) != expected or np.any(indices < 1) or np.any(indices > 1449):
            raise ValueError(f"Duplicate/out-of-range indices in {key}")
        result[name] = indices
    if set(result["train"]) & set(result["test"]) or set(np.concatenate(list(result.values()))) != set(range(1, 1450)):
        raise ValueError("Standard split must partition all 1449 labeled frames")
    return result


def evaluation_mask(depth, minimum_m=0.001, maximum_m=10.0):
    if depth.shape != NYU_SHAPE:
        raise ValueError(f"NYUv2 GT must be original {NYU_SHAPE}, got {depth.shape}")
    if not (0 < minimum_m < maximum_m):
        raise ValueError("Invalid depth interval")
    mask = np.zeros(NYU_SHAPE, dtype=bool)
    y0, y1, x0, x1 = NYU_CROP_YX
    mask[y0:y1, x0:x1] = True
    return mask & np.isfinite(depth) & (depth >= minimum_m) & (depth <= maximum_m)


@dataclass
class NYUSample:
    sample_id: str
    matlab_index: int
    rgb: np.ndarray
    depth_m: np.ndarray
    valid_gt: np.ndarray
    scene: str


class NYUv2Labeled:
    def __init__(self, path):
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        try:
            if self.file["images"].shape != (1449, 3, 640, 480) or self.file["images"].dtype != np.uint8:
                raise ValueError("Expected MATLAB images as HDF5 (N,C,W,H) uint8")
            if self.file["depths"].shape != (1449, 640, 480) or not np.issubdtype(self.file["depths"].dtype, np.floating):
                raise ValueError("Expected MATLAB depths as HDF5 (N,W,H) floating-point meters")
        except Exception:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def load(self, matlab_index):
        if not isinstance(matlab_index, (int, np.integer)) or not 1 <= matlab_index <= 1449:
            raise ValueError("Use the original 1-based MATLAB index in [1,1449]")
        index = matlab_index - 1
        # HDF5 reverses MATLAB dimensions: transpose, do not rotate/mirror or divide by 1000.
        rgb = self.file["images"][index].transpose(2, 1, 0).copy()
        depth = self.file["depths"][index].T.astype(np.float32, copy=True)
        scene = ""
        if "scenes" in self.file:
            reference = self.file["scenes"][0, index]
            scene = "".join(chr(int(c)) for c in self.file[reference][()].ravel())
        return NYUSample(f"nyuv2_{matlab_index:04d}", int(matlab_index), rgb, depth,
                         evaluation_mask(depth), scene)
