# RGB–depth edge association diagnostic

This Part 3 diagnostic tests whether RGB edges locate NYUv2 depth discontinuities before selecting an edge input for a residual network. It does not train a model or measure depth prediction accuracy.

From the repository root:

```powershell
& .\.venv-cuda\Scripts\python.exe -X utf8 scripts/part3_improvement/analyze_rgb_depth_edges.py
```

Use `--output` with a new directory to repeat a run. Existing output directories are never overwritten. The default output is `outputs/part3_improvement/edge_association/nyuv2_sobel_canny_20261118/`.

## Data and controls

- Original `data/nyuv2/nyu_depth_v2_labeled.mat` RGB and floating-point filled depth; source hashes and the official split are checked.
- The pinned zero-based IDs in `configs/edge_association_split.json` divide the official 795 training images into 577 fitting and 218 validation images. The script checks scene separation. The 654 test images are not decoded or evaluated.
- Both detectors use grayscale RGB, Gaussian smoothing (5×5, sigma 1), and 3×3 Sobel gradients. Sobel thresholds the L2 magnitude; Canny additionally applies nonmaximum suppression and hysteresis, with low/high ratio 0.4 and L2 gradient enabled.
- Thresholds are selected using fitting RGB edge density alone, for targets 1%, 2%, 5%, and 10%. GT boundary matches are not used for selection. Actual densities are reported; the Canny grid need not achieve the target exactly.
- At the 5% fitting target, an additional Sobel control keeps exactly as many edge pixels as Canny **in each image**, using RGB magnitude ranks with deterministic tie breaking. This separates edge count from spatial placement; it is not the same as deploying a fixed threshold.

## Measurement

The primary GT boundary is a four-neighbor log-depth jump greater than `log(1.1)`, retaining both endpoints. The valid domain is the course crop `[45:471, 41:601]`, finite depth strictly inside `(0.001, 10)` meters, eroded by a 3×3 kernel. Match tolerance is Euclidean distance at most 2 pixels.

Precision is the fraction of RGB edge pixels near a GT boundary; recall is the fraction of GT boundary pixels near an RGB edge. Matching is tolerance coverage, not one-to-one correspondence. Pooled and image-macro scores are saved separately. A same-count uniform random pixel reference is evaluated analytically, without Monte Carlo noise; it does not preserve the spatial structure of real contours.

Sensitivity checks use relative depth jumps 5%, 10%, and 20%, and tolerances 1, 2, and 3 pixels. The Canny versus exact-count Sobel difference has a paired bootstrap interval resampling validation scenes (3,000 replicates). This interval is conditional on the fixed fitting calibration and protocol.

## Results to inspect

- `analysis.md` and `index.html`: tables, figures, and examples.
- `per_image.csv`, `summary.json`, `summary.csv`: recomputable counts and scores.
- `calibration.json`: chosen thresholds and actual fitting density estimates.
- `paired_scene_bootstrap.json`: paired uncertainty estimates.
- `example_selection.json`: deterministic selection of high/median/low agreement and detector-difference examples; these are diagnostic examples, not a random representative sample.
- `metadata.json`: split, versions, implementation hashes, artifact hashes, scene counts, and completion status.

RGB texture, illumination, filled-depth artifacts, and registration errors can affect correspondence. An unmatched RGB edge is not automatically a texture edge, and a GT discontinuity is not automatically a semantic object boundary. This diagnostic can motivate Sobel/Canny input controls, but only a later controlled training experiment can establish whether either improves the frozen UniDepth residual model.

## Supplement: RGB and depth use both detectors

After the original diagnostic exists, run:

```powershell
& .\.venv-cuda\Scripts\python.exe -X utf8 scripts/part3_improvement/analyze_rgb_depth_edge_cross.py
```

The separate output is `outputs/part3_improvement/edge_association/nyuv2_sobel_canny_cross_20261118/`. The original output is preserved. `--source`, `--output`, `--data-root`, and `--split` can override locations.

The supplement evaluates all four RGB Sobel/Canny × GT-depth Sobel/Canny combinations. RGB retains its original fitting-calibrated 5% target setting. Depth thresholds are calibrated independently using only fitting depth-edge density (1%, 2%, 5% targets). The primary depth target is 2%, fixed before running the supplement and chosen near the original jump-edge density, not selected by validation scores. Report actual densities and any unreachable targets.

Depth Sobel operates on floating-point `log(depth in meters)` after a 5×5 Gaussian with sigma 1. Depth Canny uses the same derivatives through OpenCV's custom-gradient overload: multiply derivatives by the fixed global factor 512, round to int16, and scale thresholds identically. It uses L2 magnitude and low/high ratio 0.4. Overflow raises an error rather than clipping. This avoids per-image normalization and conversion of depth into an 8-bit or color image; derivative quantization is still an explicit approximation (step 1/512).

Invalid depth is masked during smoothing, and accepted pixels must have valid depth in a 9×9 neighborhood, covering Gaussian, derivative, and nonmaximum-suppression support. Derivatives without a fully valid 7×7 support are zeroed before Canny connectivity. All RGB methods, depth methods, and original jump references share this domain. Additional exclusions relative to the previous run are recorded. The original jump-reference comparisons are recomputed on this domain to keep them comparable.

In the additional quantity control, each image's Sobel RGB mask has exactly the RGB Canny count, and each image's Sobel depth mask has exactly the depth Canny count. RGB and depth counts are not forced to equal one another. Tolerances 1/2/3 px are evaluated. The original six diagnostic images are retained rather than choosing new favorable examples.

The CSV fields `detected_edges` and `reference_edges` refer to the first and second masks, respectively; `mode=anchor` compares either RGB or depth detections with the unchanged 10% jump definition. Depth Sobel/Canny also see gradients on smooth surfaces, so changing the reference changes the question. Diagonal F1 values cannot establish which RGB detector is more accurate, and no result in this supplement measures a learned model's depth accuracy.
