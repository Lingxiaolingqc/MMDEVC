"""RGB/GT boundary association, independent of any learned depth prediction."""

import cv2
import numpy as np
from scipy.ndimage import binary_erosion

from p3_depth.temporal.fusion import log_edge_strength


def evaluation_domain(gt):
    if gt.shape != (480, 640):
        raise ValueError('Expected original NYUv2 480x640 depth')
    region = np.zeros(gt.shape, bool)
    region[45:471, 41:601] = True
    region &= np.isfinite(gt) & (gt > .001) & (gt < 10)
    return region, binary_erosion(region, structure=np.ones((3, 3)), border_value=0)


def rgb_gradients(rgb):
    """Both detectors share uint8 grayscale, Gaussian smoothing and Sobel aperture."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 1., borderType=cv2.BORDER_REFLECT_101)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    return smooth, np.hypot(dx, dy)


def canny_mask(gray, threshold, domain):
    return (cv2.Canny(gray, .4 * threshold, threshold, apertureSize=3, L2gradient=True) > 0) & domain


def gt_edge_masks(gt, region, domain, jumps=(.05, .1, .2)):
    strength = log_edge_strength(gt, region)
    return {jump: (strength > np.log1p(jump)) & domain for jump in jumps}


def disk(radius):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    return ((yy**2 + xx**2) <= radius**2).astype(np.uint8)


def near_edges(edges, radius):
    return cv2.dilate(edges.astype(np.uint8), disk(radius)).astype(bool)


def top_count_mask(magnitude, domain, count):
    """Exactly count pixels using only RGB ranks. Stable linear index breaks ties."""
    indices = np.flatnonzero(domain)
    if not 0 <= count <= len(indices):
        raise ValueError('Count outside valid domain')
    result = np.zeros(domain.shape, bool)
    if count:
        values = magnitude.ravel()[indices]
        cut = np.partition(values, len(values)-count)[len(values)-count]
        above = indices[values > cut]
        tied = indices[values == cut]
        chosen = np.concatenate([above, tied[:count-len(above)]])
        result.ravel()[chosen] = True
    return result


def random_recall(domain_size, selected_count, neighborhood_sizes):
    """Exact expected recall for a uniform fixed-count sample without replacement."""
    probabilities = []
    for size in np.unique(neighborhood_sizes):
        if size > domain_size - selected_count:
            absent = 0.
        else:
            absent = float(np.prod([(domain_size-selected_count-i)/(domain_size-i) for i in range(int(size))]))
        probabilities.append((int(size), 1-absent))
    lookup = dict(probabilities)
    return float(np.mean([lookup[int(size)] for size in neighborhood_sizes])) if len(neighborhood_sizes) else None


def score_association(rgb_edges, gt_edges, domain, tolerance, gt_near=None, neighborhood_sizes=None):
    if any(x.shape != domain.shape for x in (rgb_edges, gt_edges)):
        raise ValueError('Mask shapes differ')
    if np.any((rgb_edges | gt_edges) & ~domain):
        raise ValueError('Edges outside evaluation domain')
    n = int(domain.sum())
    if not n:
        raise ValueError('Empty evaluation domain')
    nr, ng = int(rgb_edges.sum()), int(gt_edges.sum())
    gn = near_edges(gt_edges, tolerance) if gt_near is None else gt_near
    rn = near_edges(rgb_edges, tolerance) & domain
    mr, mg = int((rgb_edges & gn).sum()), int((gt_edges & rn).sum())
    precision = mr/nr if nr else None
    recall = mg/ng if ng else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0. if ng else None)
    random_p = float((gn & domain).sum()/n)
    if neighborhood_sizes is None:
        sizes = cv2.filter2D(domain.astype(np.float32), -1, disk(tolerance).astype(np.float32), borderType=cv2.BORDER_CONSTANT)
        neighborhood_sizes = np.rint(sizes[gt_edges]).astype(int)
    # For rectangular valid crops, nearly all neighborhoods have identical size.
    unique, counts = np.unique(neighborhood_sizes, return_counts=True)
    random_r = sum(random_recall(n, nr, np.array([k])) * c for k, c in zip(unique, counts))/ng if ng else None
    return dict(domain_pixels=n, rgb_edges=nr, gt_edges=ng, matched_rgb=mr, matched_gt=mg,
                precision=precision, recall=recall, f1=f1, density=nr/n,
                random_matched_rgb=nr*random_p, random_matched_gt=ng*random_r if ng else 0.,
                random_precision=random_p, random_recall=random_r,
                both_near_pixels=int((domain & rn & gn).sum()),
                rgb_only_near_pixels=int((domain & rn & ~gn).sum()),
                depth_only_near_pixels=int((domain & ~rn & gn).sum()),
                neither_near_pixels=int((domain & ~rn & ~gn).sum()))
