"""Read Middlebury v3 quarter-resolution training data without modifying it.

PFM conventions follow MiddEval3 SDK 1.6, ImageIO.cpp::ReadFilePFM:
use the header sign for byte order, flip rows, and preserve raw pixel disparities.
The header magnitude (commonly 0.003922) is NOT a disparity multiplier.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_SCENES = (
    "Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE",
    "Piano", "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP",
    "Recycle", "Shelves", "Teddy", "Vintage",
)
MANIFEST_FIELDS = (
    "sample_id", "left_path", "right_path", "disparity_gt_path",
    "calibration_path", "split", "notes",
)


@dataclass(frozen=True)
class Calibration:
    cam0: np.ndarray
    cam1: np.ndarray
    baseline_mm: float
    doffs_px: float
    width: int
    height: int
    ndisp: int

    @property
    def focal_px(self) -> float:
        return float(self.cam0[0, 0])


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    left: Path
    right: Path
    disparity_gt: Path
    calibration: Path

    @property
    def nonocclusion_mask(self) -> Path:
        return self.disparity_gt.with_name("mask0nocc.png")


@dataclass
class MiddleburySample:
    paths: SamplePaths
    calibration: Calibration
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    disparity_px: np.ndarray
    depth_m: np.ndarray
    valid_gt: np.ndarray
    nonoccluded: np.ndarray
    occlusion_labels: np.ndarray
    pfm_header_scale: float


def read_disparity_pfm(path: str | Path) -> tuple[np.ndarray, float]:
    """Return top-to-bottom float32 disparities and the unmodified header value."""
    path = Path(path)
    with path.open("rb") as handle:
        def header_line() -> bytes:
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Incomplete PFM header: {path}")
                value = line.strip()
                if value and not value.startswith(b"#"):
                    return value

        if header_line() != b"Pf":
            raise ValueError(f"Expected single-channel Pf disparity file: {path}")
        dimensions = header_line().split()
        if len(dimensions) != 2:
            raise ValueError(f"Invalid PFM dimensions: {path}")
        width, height = map(int, dimensions)
        scale = float(header_line())
        if width <= 0 or height <= 0 or not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid PFM dimensions or scale: {path}")
        payload = handle.read()
    if len(payload) != width * height * 4:
        raise ValueError(f"PFM payload size does not match {width}x{height}: {path}")
    dtype = "<f4" if scale < 0 else ">f4"
    bottom_up = np.frombuffer(payload, dtype=dtype).reshape(height, width)
    # Middlebury SDK preserves values even when abs(scale) != 1.
    disparity = np.flipud(bottom_up).astype(np.float32, copy=True)
    return disparity, scale


def read_calibration(path: str | Path) -> Calibration:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or key.strip() in values:
            raise ValueError(f"Malformed or duplicate calibration entry: {line}")
        values[key.strip()] = value.strip()
    required = {"cam0", "cam1", "baseline", "doffs", "width", "height", "ndisp"}
    if missing := required - values.keys():
        raise ValueError(f"Missing calibration fields: {sorted(missing)}")

    def matrix(key: str) -> np.ndarray:
        raw = values[key]
        if not (raw.startswith("[") and raw.endswith("]")):
            raise ValueError(f"Invalid {key} matrix")
        rows = [[float(x) for x in row.split()] for row in raw[1:-1].split(";")]
        result = np.asarray(rows, dtype=np.float64)
        if result.shape != (3, 3) or not np.isfinite(result).all():
            raise ValueError(f"Invalid {key} matrix")
        if result[0, 0] <= 0 or result[1, 1] <= 0:
            raise ValueError(f"Nonpositive focal length in {key}")
        if not np.allclose(result[2], [0, 0, 1]):
            raise ValueError(f"Invalid homogeneous row in {key}")
        return result

    calib = Calibration(
        cam0=matrix("cam0"), cam1=matrix("cam1"),
        baseline_mm=float(values["baseline"]), doffs_px=float(values["doffs"]),
        width=int(values["width"]), height=int(values["height"]), ndisp=int(values["ndisp"]),
    )
    if not np.isfinite(calib.baseline_mm) or calib.baseline_mm <= 0:
        raise ValueError("baseline must be a positive finite length in millimeters")
    if not np.isfinite(calib.doffs_px) or min(calib.width, calib.height, calib.ndisp) <= 0:
        raise ValueError("Invalid doffs, dimensions or disparity search bound")
    if not np.isclose(calib.cam0[0, 0], calib.cam1[0, 0], rtol=1e-6):
        raise ValueError("Rectified focal lengths differ")
    expected_doffs = calib.cam1[0, 2] - calib.cam0[0, 2]
    if not np.isclose(calib.doffs_px, expected_doffs, atol=0.01, rtol=0):
        raise ValueError("doffs disagrees with the principal-point difference")
    return calib


def disparity_to_depth_m(disparity_px: np.ndarray, calib: Calibration) -> np.ndarray:
    """Convert pixel disparity to camera-axis Z in meters, keeping invalids NaN."""
    disparity = np.asarray(disparity_px, dtype=np.float64)
    if disparity.ndim != 2:
        raise ValueError("Disparity must be a two-dimensional H x W array")
    denominator = disparity + calib.doffs_px
    valid = np.isfinite(disparity) & np.isfinite(denominator) & (denominator > 0)
    result = np.full(disparity.shape, np.nan, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(calib.focal_px * (calib.baseline_mm / 1000), denominator,
                  out=result, where=valid)
        depth = result.astype(np.float32)
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


def discover_samples(data_root: str | Path) -> list[SamplePaths]:
    """Pair trainingQ inputs and GT by scene, including separately unpacked ZIPs.

    Duplicate scene roots, missing pairs and missing files are errors. testQ,
    trainingH and trainingF are deliberately excluded.
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Middlebury root does not exist: {root}")

    def collect(filename: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for path in sorted(root.rglob(filename)):
            if path.parent.parent.name != "trainingQ":
                continue
            scene = path.parent.name
            if scene in result:
                raise ValueError(f"Duplicate trainingQ {filename} for {scene}")
            result[scene] = path
        return result

    lefts, disparities = collect("im0.png"), collect("disp0GT.pfm")
    if not lefts:
        raise ValueError("No trainingQ/im0.png files found under the configured root")
    if lefts.keys() != disparities.keys():
        raise ValueError(f"Input/GT scene mismatch: missing GT={sorted(lefts.keys()-disparities.keys())}; "
                         f"missing inputs={sorted(disparities.keys()-lefts.keys())}")
    samples = []
    for scene, left in sorted(lefts.items()):
        sample = SamplePaths(scene, left, left.with_name("im1.png"),
                             disparities[scene], left.with_name("calib.txt"))
        for path in (sample.left, sample.right, sample.disparity_gt,
                     sample.calibration, sample.nonocclusion_mask):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {scene} file: {path}")
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Sample path resolves outside the data root: {path}")
        samples.append(sample)
    return samples


def load_sample(paths: SamplePaths) -> MiddleburySample:
    calib = read_calibration(paths.calibration)
    with Image.open(paths.left) as image:
        left = np.array(image.convert("RGB"))
    with Image.open(paths.right) as image:
        right = np.array(image.convert("RGB"))
    with Image.open(paths.nonocclusion_mask) as image:
        labels = np.array(image)
    disparity, scale = read_disparity_pfm(paths.disparity_gt)
    shape = (calib.height, calib.width)
    if left.shape != (*shape, 3) or right.shape != (*shape, 3):
        raise ValueError(f"RGB dimensions do not match calibration: {paths.sample_id}")
    if disparity.shape != shape or labels.shape != shape:
        raise ValueError(f"GT or mask dimensions do not match RGB: {paths.sample_id}")
    if not np.isin(labels, [0, 128, 255]).all():
        raise ValueError(f"Unexpected mask labels: {paths.sample_id}")
    # Official mask: 0 unknown, 128 occluded, 255 nonoccluded (not > 0).
    if not np.array_equal(np.isfinite(disparity), labels != 0):
        raise ValueError(f"PFM validity disagrees with mask; check file pairing/orientation: {paths.sample_id}")
    depth = disparity_to_depth_m(disparity, calib)
    valid_gt = (labels != 0) & np.isfinite(depth)
    if np.any((labels != 0) & ~np.isfinite(depth)) or not valid_gt.any():
        raise ValueError(f"Known GT has nonphysical or empty depth: {paths.sample_id}")
    return MiddleburySample(paths, calib, left, right, disparity, depth, valid_gt,
                            labels == 255, labels, scale)


def write_manifest(samples: list[SamplePaths], data_root: str | Path, destination: str | Path) -> None:
    """Refresh paths, preserving existing split/notes and refusing stale/duplicate IDs."""
    root, destination = Path(data_root).resolve(), Path(destination)
    previous: dict[str, dict[str, str]] = {}
    if destination.exists():
        with destination.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(MANIFEST_FIELDS):
                raise ValueError("Unexpected existing Middlebury manifest header")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed Middlebury manifest row")
                sample_id = row["sample_id"]
                if sample_id in previous:
                    raise ValueError(f"Duplicate manifest ID: {sample_id}")
                if row["split"] not in {"unassigned", "dev", "evaluation"}:
                    raise ValueError(f"Unsupported split for {sample_id}: {row['split']}")
                previous[sample_id] = row
    ids = {sample.sample_id for sample in samples}
    if len(ids) != len(samples) or previous.keys() - ids:
        raise ValueError("Duplicate discovered IDs or existing manifest IDs absent from the data")
    rows = []
    for sample in samples:
        old = previous.get(sample.sample_id, {})
        rows.append({
            "sample_id": sample.sample_id,
            "left_path": sample.left.resolve().relative_to(root).as_posix(),
            "right_path": sample.right.resolve().relative_to(root).as_posix(),
            "disparity_gt_path": sample.disparity_gt.resolve().relative_to(root).as_posix(),
            "calibration_path": sample.calibration.resolve().relative_to(root).as_posix(),
            "split": old.get("split", "unassigned"),
            "notes": old.get("notes", "official trainingQ; assign dev/evaluation before tuning"),
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
