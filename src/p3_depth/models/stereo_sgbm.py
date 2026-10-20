"""CPU StereoSGBM with signed right disparity and explicit left/right rejection."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from p3_depth.data.middlebury import Calibration, disparity_to_depth_m


@dataclass(frozen=True)
class StereoConfig:
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    pre_filter_cap: int = 31
    lr_max_diff_px: float = 1.0

    def __post_init__(self):
        for key in ("block_size", "uniqueness_ratio", "speckle_window_size",
                    "speckle_range", "pre_filter_cap"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.block_size < 1 or self.block_size % 2 != 1:
            raise ValueError("block_size must be positive and odd")
        if min(self.uniqueness_ratio, self.speckle_window_size, self.speckle_range) < 0:
            raise ValueError("Uniqueness and speckle parameters must be nonnegative")
        if self.pre_filter_cap <= 0:
            raise ValueError("pre_filter_cap must be positive")
        if not np.isfinite(self.lr_max_diff_px) or self.lr_max_diff_px < 0:
            raise ValueError("lr_max_diff_px must be finite and nonnegative")

    def resolved(self, calibration: Calibration) -> dict:
        # Only the supplied conservative search bound is used; never GT min/max.
        count = ((calibration.ndisp + 15) // 16) * 16
        if not 0 < count < min(calibration.width, 2048):
            raise ValueError("Disparity search must fit the image and int16 fixed-point output")
        if self.block_size > min(calibration.width, calibration.height):
            raise ValueError("block_size exceeds image dimensions")
        return {
            **asdict(self), "min_disparity": 0, "num_disparities": count,
            "right_min_disparity": 1 - count,
            "p1": 8 * self.block_size ** 2, "p2": 32 * self.block_size ** 2,
            "mode": "SGBM_3WAY", "input": "RGB uint8 -> grayscale uint8",
            "disp12_max_diff": 1, "search_source": "ceil(calibration.ndisp / 16) * 16",
            "lr_sampling": "nearest x: floor(x - d_left + 0.5); signed d_right",
        }


@dataclass
class StereoPrediction:
    left_fixed16: np.ndarray
    right_fixed16: np.ndarray
    left_disparity_px: np.ndarray
    right_disparity_px: np.ndarray
    lr_error_px: np.ndarray
    disparity_px: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray
    parameters: dict


def decode_disparity(raw: np.ndarray, min_disparity: int) -> np.ndarray:
    """OpenCV int16 is disparity * 16; (minDisparity - 1) * 16 is invalid."""
    if raw.dtype != np.int16 or raw.ndim != 2:
        raise ValueError("Expected an H x W int16 StereoSGBM output")
    result = raw.astype(np.float32) / 16.0
    result[raw <= (min_disparity - 1) * 16] = np.nan
    return result


def left_right_consistency(left: np.ndarray, right: np.ndarray,
                           max_diff_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Return left-view mask and |dL(x) + dR(x-dL(x))|; unavailable residuals are NaN."""
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("Left and right disparities must have identical H x W shape")
    if not np.isfinite(max_diff_px) or max_diff_px < 0:
        raise ValueError("max_diff_px must be finite and nonnegative")
    y, x = np.indices(left.shape)
    finite_left = np.isfinite(left)
    xr = x - np.where(finite_left, left, 0)
    inside = finite_left & (xr >= 0) & (xr <= left.shape[1] - 1)
    nearest = np.floor(np.clip(xr, 0, left.shape[1] - 1) + 0.5).astype(np.intp)
    sampled_right = right[y, nearest]
    comparable = inside & np.isfinite(sampled_right)
    residual = np.full(left.shape, np.nan, dtype=np.float32)
    residual[comparable] = np.abs(left[comparable] + sampled_right[comparable])
    return comparable & (residual <= max_diff_px), residual


def predict_stereo(left_rgb: np.ndarray, right_rgb: np.ndarray,
                   calibration: Calibration, config: StereoConfig) -> StereoPrediction:
    """Match original-size rectified RGB images. This API accepts no GT or GT mask."""
    shape = (calibration.height, calibration.width, 3)
    for image in (left_rgb, right_rgb):
        if image.shape != shape or image.dtype != np.uint8:
            raise ValueError(f"Expected RGB uint8 images of calibrated shape {shape}")
    params = config.resolved(calibration)
    gray_l = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)

    def matcher(minimum):
        return cv2.StereoSGBM_create(
            minDisparity=minimum, numDisparities=params["num_disparities"],
            blockSize=config.block_size, P1=params["p1"], P2=params["p2"],
            disp12MaxDiff=params["disp12_max_diff"], preFilterCap=config.pre_filter_cap,
            uniquenessRatio=config.uniqueness_ratio,
            speckleWindowSize=config.speckle_window_size, speckleRange=config.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    raw_l = matcher(params["min_disparity"]).compute(gray_l, gray_r)
    raw_r = matcher(params["right_min_disparity"]).compute(gray_r, gray_l)
    left = decode_disparity(raw_l, params["min_disparity"])
    right = decode_disparity(raw_r, params["right_min_disparity"])
    consistent, residual = left_right_consistency(left, right, config.lr_max_diff_px)
    disparity = np.where(consistent, left, np.nan).astype(np.float32)
    depth = disparity_to_depth_m(disparity, calibration)
    valid = consistent & np.isfinite(depth)
    disparity[~valid] = np.nan
    return StereoPrediction(raw_l, raw_r, left, right, residual, disparity, depth, valid, params)
