# Independent Direction A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute in this session unless the user requests delegation. This document is proposed for review; it does not claim implementation or experiments are complete.

**Goal:** Independently complete the revised assignment and test whether flow-guided fusion improves video stability while preserving moving boundaries and responsiveness.

**Architecture:** Frozen official Small depth models feed separate reproducible metric-image and video workflows. Cache raw frame-wise predictions and flow, then implement our own backward warping, occlusion rejection, and confidence fusion. A fixed evaluator and controlled visualizations compare our own unchanged runs with each ablation.

**Tech Stack:** Python 3.11 as the initial compatibility target, NumPy, OpenCV, PyTorch/torchvision, h5py, SciPy for the official split MAT, pytest, Matplotlib; official DA V2, VDA, and UniDepth repositories/checkpoints pinned after a one-image compatibility check. Local RTX 4070 Laptop GPU, 8 GB VRAM; optional course GPU only for measured constraints.

**Spec:** [Proposed design](../specs/2026-11-08-direction-a-design.md).

## Global Constraints

- All tracked contributions belong under `independent/`.
- Work on `feat/independent-*`, `fix/independent-*`, `docs/independent-*`, or `experiment/independent-*` branches.
- Never commit or push to `main` or `master`. Never force-push shared history or merge without the user's instruction.
- Do not inspect, adapt, or import the groupmate's implementation, configurations, results, or selected parameters.
- Follow the revised Small-model rule. Start locally on the RTX 4070 Laptop GPU, 8 GB VRAM. Server use is optional and provides no documented assessment bonus.
- NYU: 795 development/654 test; resize to 480 x 640; crop `[45:471, 41:601]`; finite GT strictly greater than 0.001 and strictly less than 10 m; predictions clipped to `[0.001, 10]`; equal per-image averages; no test-set scale alignment. The endpoint convention is recorded explicitly because the PDF says "between".
- Use the same real first 100 frames at original FPS. Short source clips are a documented submission blocker, not permission to invent frames.
- Never normalize each video frame independently. Fusion displays use the unchanged method's fixed whole-clip range.
- No performance claim is accepted without our own reproducible run. Video proxies do not establish metric accuracy.

## Review Focus

- Short/corrupt or variable-rate video: count successfully decoded frames, retain source timestamps, and expose an unmet frame requirement (Task 1).
- Geometry errors: wrong warp direction, component scaling, and border sampling must fail identity/translation tests (Task 4).
- Empty masks, NaNs, flat depth, scene cuts, or unreliable flow: use explicit errors/current-frame fallback rather than unstable history (Tasks 3-5).
- Evaluation leakage and degenerate smooth predictions: candidate-independent masks, frozen settings, and ground-truth synthetic controls (Task 6).
- Stale caches and misleading timing: hash complete inference configurations; distinguish replay from full-pipeline performance (Tasks 2 and 6).

## File structure after implementation

All paths below are relative to `independent/`. Only planning documents and Git safeguards exist now; this is the intended implementation layout.

```text
pyproject.toml                 our package/CLI and minimal dependencies
requirements.lock.txt          tested environment versions
configs/                       baseline and frozen experiment configs
manifests/                     data/checkpoint provenance and split IDs
src/depth_project/
  records.py                   shared typed records and metadata
  video.py                     decode, inspection, frame selection
  data.py                      official NYU and Middlebury readers
  stereo.py                    SGBM and disparity-to-depth conversion
  models.py                    thin official pretrained-model adapters
  cache.py                     prediction/flow provenance and reuse
  metrics.py                   exact NYU/stereo metrics and diagnostics
  flow.py                      Farneback and optional RAFT adapter
  warp.py                      backward gather and validity masks
  fusion.py                    explicit history state and weighting
  render.py                    fixed normalization and comparison videos
  application.py               simple depth-aware background blur
  cli.py                       inspect, infer, evaluate, fuse, render commands
tests/                         numerical and data-contract checks
docs/learning/                 small worked examples and user's explanations
results/                       small summaries/selected report figures
report/                        CVPR source and bibliography
data/, checkpoints/, third_party/, runs/, outputs/   ignored large artifacts
```

## Task 1: Video inspection, provenance, and a clean local environment

**Files:** `pyproject.toml`, `src/depth_project/{records,video,cli}.py`, `tests/test_video.py`, `manifests/sources.json`, `docs/learning/01-video.md`.

**Interfaces:** `VideoInfo` records source SHA-256, dimensions, nominal FPS, decoded frame count, timestamps when available, requested/available frame counts, and compliance status. `inspect_video(path: Path, required_frames: int = 100) -> VideoInfo`; `iter_rgb_frames(path: Path, limit: int | None) -> Iterator[Frame]`, where `Frame` has zero-based index, timestamp seconds, and HWC uint8 RGB.

- [ ] Work through one BGR pixel becoming RGB; inspect the first three real frames with the user.
- [ ] Create a project-local `.venv`; verify CUDA-compatible torch/torchvision installation and one basic CUDA operation. Record versions. If Windows upstream compatibility fails, use an isolated WSL2/Linux environment; do not change the system Python or install every model's requirements into a shared environment blindly.
- [ ] Test `test_counts_decoded_frames_not_metadata`: a known 3-frame fixture produces count 3 and indices `[0,1,2]`; requesting 100 marks it short. Test `test_corrupt_video_is_error` and a known red/blue RGB round-trip with codec tolerance.
- [ ] Run `python -m pytest tests/test_video.py -q`, observe failure, then implement the decoder and manifest records. Nominal frame-count metadata is diagnostic only.
- [ ] Inspect the official rollercoaster asset, hash it, and save the first three frames. Resolve any 70-versus-100 discrepancy from the actual source; no frame fabrication. Extract official download URLs from PDF hyperlinks and pin official source revisions.
- [ ] Verify with `python -m depth_project.cli inspect-video --input <official-clip> --required-frames 100 --preview-frames 3`. A short source remains visibly noncompliant while local development can proceed.
- [ ] Commit this independently testable stage; record what the user can explain.

## Task 2: One-frame depth inference, then cached Small-model video baselines

**Files:** `src/depth_project/{models,cache,render,cli}.py`, `configs/baselines.json`, `tests/test_model_contracts.py`, `tests/test_cache.py`, `docs/learning/02-depth.md`.

**Interfaces:** `DepthPrediction` has HW float32 values, `kind` (`relative` or `metric_m`), output shape, model/checkpoint identity, preprocessing identity, and precision. `predict_rgb(rgb: np.ndarray) -> DepthPrediction` for image adapters. `infer_video(frames: Sequence[Frame], config: dict) -> Iterator[DepthPrediction]` preserves the official VDA offline temporal context. `cache_key(source_sha256: str, inference_config: dict) -> str`; `compute_display_range(raw_sequence: Sequence[np.ndarray]) -> tuple[float,float]` uses whole-sequence 2nd/98th percentiles.

- [ ] Explain one frame's input, frozen pretrained weights, relative output, and missing video ground truth before batch inference.
- [ ] Test adapters with tiny local fakes: RGB HWC input, correct shape, explicit units, finite float32 output. Test cache-key changes for model hash, source hash, size, precision, preprocessing revision, frame selection, and upstream commit; reject truncated caches.
- [ ] Run the relevant tests to observe failure, implement thin official adapters/cache metadata, then rerun to passing. Keep vendor model internals external and attributed; smoke-test actual official Small checkpoints on one frame before a full clip.
- [ ] Cache our unchanged DA V2-Small predictions. Run VDA-Small offline with the revised flags. Do not treat standalone single-frame calls as equivalent to VDA's offline video inference.
- [ ] Check original-rate frame count/order, dimensions, finite outputs, whole-clip display ranges, and saturation fraction. VDA may have its own fixed range; all DA V2 fusion variants later inherit DA V2's range.
- [ ] Record baseline timing and peak memory with one model resident at a time. Depth inference FP16 is used only where supported/validated; accumulate fusion and numerical evaluation in float32/float64 as appropriate.
- [ ] Commit adapter/config/manifest code and a small provenance summary; leave model weights and full predictions ignored.

## Task 3: Independent mandatory stereo and NYU evaluation

**Files:** `src/depth_project/{data,stereo,metrics,cli}.py`, `configs/nyu-course.json`, `tests/test_data.py`, `tests/test_stereo.py`, `tests/test_metrics.py`, `results/baseline-summary.json`, `docs/learning/03-geometry-metrics.md`.

**Interfaces:** `load_nyu_sample(index: int) -> tuple[np.ndarray,np.ndarray]`; `load_nyu_split(path: Path) -> dict[str,np.ndarray]` returns zero-based disjoint development/test IDs. `disparity_to_meters(disp_px: np.ndarray, focal_px: float, baseline_mm: float, doffs: float) -> np.ndarray`; `depth_metrics(pred_valid: np.ndarray, gt_valid: np.ndarray) -> dict[str,float]` computes metrics on valid vectors. `evaluate_nyu(pred_m: np.ndarray, gt_m: np.ndarray) -> dict[str,float]` applies resizing/crop/mask and returns per-image AbsRel/RMSE/delta1 and valid count.

- [ ] Trace a stereo example: `f=1000`, `B=100 mm`, OpenCV raw disparity `160`, `doffs=2` gives `d=10 px` and `z=8.333333 m`. Distinguish this calibration-based depth from monocular relative output.
- [ ] Write exact tests: fixed-point division by 16; calibration offset and mm-to-m conversion; invalid/negative disparity; PFM endianness/orientation; different crop errors; strict depth endpoints; HDF5 axes and MATLAB indices; non-finite predictions raise an error.
- [ ] Pin the NYU metric arithmetic: GT `[1,2]`, prediction `[1,2.4]` gives AbsRel `0.1`, RMSE `sqrt(0.08)`, and delta1 `1.0`; exact `1.25` ratios fail delta1's strict threshold. Verify equal image weighting with unequal valid-pixel counts. Separate resizing from this tiny-vector metric check.
- [ ] Run those tests to failure, implement the readers, conversion, and metric helpers, then rerun. For stereo, denominator for coverage is finite positive GT; disparity MAE uses its intersection with valid predictions; always report both counts.
- [ ] Independently run all 15 trainQ pairs and UniDepthV2 ViT-S RGB-only inference on all 654 NYU test images using the exact course protocol. Use the 795 development images for any debugging/tuning. Do not combine unordered NYU images into video fusion input.
- [ ] Save per-image/per-pair metrics, IDs, hashes, configuration, and our own aggregates. Label these as course-protocol results, without claiming equality to a paper's benchmark score.
- [ ] Commit evaluation code, locked protocol, small summaries, and learning notes.

## Task 4: Motion estimation and correct backward warping

**Files:** `src/depth_project/{flow,warp}.py`, `tests/test_flow.py`, `tests/test_warp.py`, `docs/learning/04-flow-warp.md`.

**Interfaces:** `estimate_pair(prev_rgb: np.ndarray, curr_rgb: np.ndarray, config: dict) -> tuple[np.ndarray,np.ndarray]` returns full-output-resolution forward and backward HW2 pixel displacement. `warp_previous(previous: np.ndarray, backward: np.ndarray) -> tuple[np.ndarray,np.ndarray]` returns sampled values and HW boolean validity. `flow_consistency(forward: np.ndarray, backward: np.ndarray, config: dict) -> np.ndarray` returns a current-frame mask.

- [ ] Move a 1D toy feature right by one pixel, then trace why the current location samples the previous location with a negative backward displacement. Show a small arrow diagram.
- [ ] Write `test_identity_warp`, `test_one_pixel_translation`, `test_fractional_bilinear_sample`, `test_border_and_nan_are_invalid`, `test_resize_scales_xy_independently`, and forward/backward consistency tests. Known translation must align exactly on supported integer pixels; disoccluded borders must be invalid.
- [ ] Run tests to failure; implement Farneback, explicit flow-size conversion, OpenCV remap, and consistency masks; rerun tests.
- [ ] Visualize flow arrows, warped history, and rejection masks on a few development pairs. Confirm behavior at moving edges and out-of-frame coordinates before fusion.
- [ ] Commit the correct geometric primitive and its worked example. Add RAFT-Small only after Task 6's first Farneback sweep; retain this same flow contract.

## Task 5: Transparent fusion ablations and failure handling

**Files:** `src/depth_project/fusion.py`, `configs/fusion-dev.json`, `tests/test_fusion.py`, `tests/test_synthetic_sequences.py`, `docs/learning/05-fusion.md`.

**Interfaces:** `FusionState` stores previous RGB, fused relative prediction, and frame ID. `fuse_step(current: np.ndarray, aligned_previous: np.ndarray, valid: np.ndarray, confidence: np.ndarray, history_cap: float) -> np.ndarray`; `update(frame: Frame, prediction: DepthPrediction, state: FusionState | None, config: dict) -> tuple[DepthPrediction,FusionState,dict]` returns output, new state, and diagnostics.

- [ ] Hand-compute current 10/history 12/history weight 0.25 -> 10.5; invalid history -> 10. Explain how an aggressive history weight can create delayed edges.
- [ ] Write tests for the hand calculation; history cap zero gives an unchanged baseline; invalid history/current scene cut uses current; weights stay in `[0,history_cap]`; constant/flat depth has finite confidence; shuffled frame IDs are rejected; no future frame is consumed.
- [ ] Add known-truth synthetic cases: static noisy plane, moving sharp foreground, disocclusion, true depth change, and scene cut. Assert exact reset behavior and compute ghost/lag/noise diagnostics without hardcoding an unmeasured real-video improvement.
- [ ] Observe failure, implement in stages, rerun checks: unchanged; naive EMA; flow fixed weight; consistency-gated fusion; adaptive confidence. Save individual masks/weights for inspection. Initialize/reset state at every clip boundary.
- [ ] Keep relative affine history alignment optional and off. If development evidence motivates it, fit only on reliable predictions; test degenerate/insufficient support, reject nonpositive/unstable fits, and evaluate a separate ablation. Never use video or NYU test GT to fit it.
- [ ] Commit each meaningful working ablation with its configuration and learning explanation.

## Task 6: Controlled evaluation, bounded search, and a frozen final method

**Files:** `src/depth_project/{metrics,cache,cli}.py`, `configs/{sweep-dev,final}.json`, `tests/test_temporal_evaluation.py`, `results/{ablations,runtime,failures}.csv`, `docs/experiments.md`.

**Interfaces:** `evaluate_video(predictions: Sequence[np.ndarray], evaluation_flow: Sequence[np.ndarray], evaluation_masks: Sequence[np.ndarray], reference_scale: float) -> dict` uses fixed candidate-independent support/scale. `benchmark_pipeline(config: dict, repeats: int = 3) -> dict` returns stage and end-to-end timing/memory with explicit cache and I/O mode. `run_experiment(config_path: Path) -> Path` writes a complete provenance manifest and summaries.

- [ ] Test evaluation-mask identity across candidates, pair counts, empty-mask diagnostics, finite support counts, fixed reference normalization, and rejection of prediction/frame-ID mismatch. A frozen-depth negative control demonstrates why temporal error alone cannot certify quality.
- [ ] Implement and verify synthetic GT noise/edge error and settling-delay diagnostics, whole-clip visualization, and optional real-video temporal proxies. Use a frozen evaluation-flow configuration separate from candidate confidence; report flow-related bias. True 3D depth change remains a limitation of flow-aligned scalar differences.
- [ ] Run at most the 12 planned Farneback development configurations, retaining 2-3 candidates on the stability/edge/lag/runtime tradeoff. Prefer the simpler candidate if evidence does not justify added confidence modules.
- [ ] Profile the bottleneck. Optionally compare torchvision RAFT-Small using the same fusion settings and output resolution. Record its pretrained weights, input transform, padding, iterations, and flow scaling; run models sequentially. Preserve the required Small depth models.
- [ ] Warm up, synchronize CUDA, and repeat full-pipeline measurements three times. Report median and spread; distinguish full inference from cached fusion replay and decode/encode from compute. Reset/record peak allocated and reserved GPU memory for each measured mode.
- [ ] Freeze final settings and commit `configs/final.json` before final rollercoaster evaluation. Run the baseline, each selected ablation, and VDA reference on the identical genuine frames. If fewer than 100 source frames remain unresolved, retain a visible compliance warning.
- [ ] Save representative success/failure frames by fixed selection rules and disclose any manual choice. Report our own results, including negative findings and runtime costs; no copied historical scores.

## Task 7: Application, report evidence, and reproducible submission

**Files:** `src/depth_project/{application,render,cli}.py`, `tests/test_render.py`, `tests/test_submission.py`, `report/{main.tex,references.bib}`, `README.md`, `results/figures/`.

**Interfaces:** `render_comparison(frame_ids: Sequence[int], methods: dict, ranges: dict, fps: float, output: Path) -> None`; `background_blur(rgb: np.ndarray, relative_depth: np.ndarray, config: dict) -> np.ndarray`; `build_submission(run_dir: Path, output_zip: Path) -> dict` returns verified file inventory and hashes.

- [ ] Test that a depth value maps to the same color across frames and that fusion inherits the unchanged baseline range; flat maps and saturated values must render deterministically.
- [ ] Implement a simple background-blur demo with identical application settings for unchanged and fused depth. Label the relative-depth near/far convention; use the application to expose boundary bleeding rather than hiding it with a complex UI.
- [ ] Produce required RGB/DA V2/VDA comparison plus selected fusion, three real difficult cases, flow/mask illustration, ablation table, latency/memory table, and a pipeline diagram. Explain whether the adaptive method actually earns its complexity.
- [ ] Write 6-8 CVPR pages excluding references, cite the specified reading/data/models, include the public GitHub link at the end of the abstract, and state the actual independent contribution and limitations.
- [ ] Verify archive inventory, decodable videos, true frame counts, dimensions, FPS, preserved float predictions outside colorized outputs, and a fresh documented command. Keep weights/datasets/secrets out of the archive and Git history.
- [ ] Commit reviewable evidence and submission instructions on the feature branch. Publish the named feature branch and open an integration PR when requested; do not merge into main automatically.

## Priority and stopping rules

1. Complete data verification and all mandatory Small-model baselines before polishing the improvement.
2. Correct geometry and honest visual controls precede parameter search.
3. First optimize reuse and the measured bottleneck; extra compute must resolve a concrete question.
4. Stop adding modules when the controlled evidence does not show a useful benefit. A simpler fusion winning is a valid finding.
5. Complete the report/demo/reproduction path before optional extra clips, RAFT comparisons, or deployment work consume the schedule.

## Plan self-review

- Every mandatory revised requirement maps to Tasks 1-3 or 7; Direction A and its evaluation map to Tasks 4-6.
- Shared records and flow/depth conventions are defined before downstream use; official model preprocessing remains model-specific.
- The five Review Focus risks have explicit owning tasks and verification steps.
- Open dependencies: obtain actual official inputs/checkpoints, independently verify real source frame count, and validate the isolated CUDA/upstream environment. Course GPU access is not a prerequisite.
- No product source, trained model, experiment result, or performance improvement is claimed by this planning document.
