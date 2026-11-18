"""Depth Sobel and Canny on fixed-scale log-meter derivatives."""
import cv2
import numpy as np
from scipy.ndimage import binary_erosion

from .rgb_depth_edges import evaluation_domain

DERIVATIVE_SCALE = 512.


def depth_gradients(depth):
    """No per-image range scaling, colormap, or 8-bit depth quantization."""
    valid = np.isfinite(depth) & (depth > .001) & (depth < 10.)
    region, old_domain = evaluation_domain(depth)
    # Gaussian radius 2 + derivative radius 1, then one more for Canny NMS.
    derivative_valid = binary_erosion(valid, np.ones((7, 7)), border_value=0)
    domain = old_domain & binary_erosion(valid, np.ones((9, 9)), border_value=0)
    log_depth = np.log(np.where(valid, depth, 1.)).astype(np.float32)
    # Invalid neighbors cannot contaminate accepted derivatives. Normalization
    # also avoids inventing a jump to a chosen fill value outside that domain.
    weight = cv2.GaussianBlur(valid.astype(np.float32), (5, 5), 1.)
    smooth = cv2.GaussianBlur(log_depth, (5, 5), 1.) / np.maximum(weight, 1e-8)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    dx[~derivative_valid] = 0
    dy[~derivative_valid] = 0
    derivatives = []
    for values in (dx, dy):
        scaled = np.rint(values * DERIVATIVE_SCALE)
        if np.max(np.abs(scaled)) > 32767:
            raise ValueError('Canny derivative overflow; do not silently clip')
        derivatives.append(scaled.astype(np.int16))
    return region, domain, np.hypot(dx, dy), derivatives


def depth_canny(derivatives, high, domain):
    """high is in log-depth Sobel units; integer derivatives use global x512."""
    dx, dy = derivatives
    return (cv2.Canny(dx, dy, .4*high*DERIVATIVE_SCALE,
                     high*DERIVATIVE_SCALE, L2gradient=True) > 0) & domain
