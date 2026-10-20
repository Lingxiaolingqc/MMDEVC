"""Fixed-range single-pair diagnostics; display ranges never affect predictions."""

import numpy as np


DISPLAY_RANGES = {"disparity_px": [0, 80], "depth_m": [0, 4],
                  "absolute_depth_error_m": [0, 0.2], "lr_error_px": [0, 2]}


def save_stereo_preview(sample, prediction, destination, ranges):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e5e5")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)

    def scalar(axis, values, title, limits, unit):
        image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap,
                            vmin=limits[0], vmax=limits[1], interpolation="nearest")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label=unit, shrink=0.78)

    axes[0, 0].imshow(sample.left_rgb)
    axes[0, 0].set_title("Left RGB")
    scalar(axes[0, 1], sample.disparity_px, "GT disparity", ranges["disparity_px"], "px")
    scalar(axes[0, 2], prediction.disparity_px, "SGBM disparity after LR check", ranges["disparity_px"], "px")
    axes[0, 3].imshow(prediction.valid, cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title(f"Valid prediction: {prediction.valid.mean():.2%} of image")
    scalar(axes[1, 0], sample.depth_m, "GT-derived Z depth", ranges["depth_m"], "m")
    scalar(axes[1, 1], prediction.depth_m, "SGBM Z depth (no alignment)", ranges["depth_m"], "m")
    error = np.where(sample.valid_gt & prediction.valid,
                     np.abs(prediction.depth_m - sample.depth_m), np.nan)
    scalar(axes[1, 2], error, "Absolute depth error (valid only)", ranges["absolute_depth_error_m"], "m")
    scalar(axes[1, 3], prediction.lr_error_px, "LR residual before rejection", ranges["lr_error_px"], "px")
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(f"{sample.paths.sample_id} | StereoSGBM development baseline | "
                 "gray = invalid/unavailable; display limits recorded in metadata")
    fig.savefig(destination, dpi=130)
    plt.close(fig)
