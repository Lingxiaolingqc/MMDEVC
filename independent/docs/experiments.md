# Measured experiments and deliverables

## Final metric improvement on 2026-11-12

The selected final improvement is development-calibrated RGB-guided residual refinement of frozen UniDepthV2 ViT-S. The 28 simple candidates and a 388,001-parameter learned refiner were selected using a whole-scene development split of 577 fitting and 218 validation images. Best checkpoint epoch 30 and configuration/weight digest were committed at ad3b536 before the 654-image test run.

| Method | AbsRel | RMSE (m) | Delta1 |
| --- | ---: | ---: | ---: |
| Unchanged UniDepthV2-S | 0.11455 | 0.39889 | 0.89222 |
| Global scale | 0.10151 | 0.39484 | 0.89714 |
| Flip averaging | 0.11427 | 0.39640 | 0.89432 |
| Selected flip + scale | 0.10109 | 0.39251 | 0.89939 |
| Learned residual refinement | **0.09564** | **0.37080** | **0.91374** |

This is a 16.50% reduction in actual metric-depth AbsRel, with a scene-cluster bootstrap 95% interval of 12.89%-19.94%. 431 images improve and 223 worsen. Median synchronized compute rises from 48.8 to 127.3 ms/image, with about 512 MiB peak allocated. Baseline predictions reproduce exactly, with maximum pixel discrepancy zero. The unchanged foundation input/checkpoint and exact course output/crop/masks/aggregation are shared. This is supervised NYU development adaptation. It establishes improvement over the named course reference under this setting; broader published SOTA comparisons remain unmeasured.

See [complete metric results](../results/metric-improvement.json), [paired image scores](../results/metric-residual_refiner-per-image.csv), [development search](../results/metric-development-search.json), [training curve records](../results/metric-refiner-training.json), and [GT/error comparisons](../results/figures/metric-comparison.png). Figure rows use largest gain, median change, and largest regression. Metric application blur uses a manually chosen shared 3 m focus threshold and 0.4 m transition, showing near/far regions in the same selected example.

The remaining sections preserve the initial required baselines and temporal study.

All results below are our own local runs on an RTX 4070 Laptop GPU. The [execution ledger](execution-ledger.md) records configuration freezes, review findings, fixes, and decisions. The [README](../README.md) gives setup and reproduction commands.

## Static baselines

| Evaluation | Measured result | Evidence |
| --- | --- | --- |
| Middlebury, all 15 quarter-resolution pairs | Mean pair disparity MAE 1.89470 px; mean valid coverage 75.689% | [Per-pair results and settings](../results/stereo-baseline.json) |
| NYU, all 654 test images | AbsRel 0.114546; RMSE 0.398888 m; delta1 0.892218 | [Summary](../results/nyu-baseline.json), [654 per-image rows](../results/nyu-per-image.csv) |

NYU uses the exact course crop, masks, equal image weighting, and raw metric scale. No scale fitting is used. Stereo coverage is reported because removing difficult pixels can lower error.

## Frozen video study

Selection used only the first 100 Tokyo development frames. The 12-candidate search selected the adaptive history cap 0.75, consistency beta 0.5, and depth-agreement scale 0.5. Primary configuration commit f35e63c predates its rollercoaster fusion evaluation.

| Method | Temporal proxy | Baseline-edge strength | Synthetic step settling |
| --- | ---: | ---: | ---: |
| Unchanged DA V2-Small | 0.10797 | 100.0% | 1 frame |
| Unaligned EMA | 0.12938 | 30.3% | 9 frames |
| Fixed flow fusion | 0.03688 | 76.6% | 9 frames |
| Flow + occlusion | 0.03461 | 79.4% | 9 frames |
| Adaptive confidence | 0.08829 | 97.6% | 1 frame |

The selected method reduces this stability proxy by 18.23%, while preserving 97.58% of baseline-edge strength. A lower proxy alone cannot establish better depth: the stronger smoothing rows lose boundaries and respond slowly. There is no video depth ground truth. See [the complete primary evaluation](../results/final-evaluation.json).

The [secondary flow-resolution study](../results/flow-resolution-study.json) was performed after the primary evaluation. Width 320 was chosen using a declared development-only rule and committed before its secondary evaluation. It retains a 16.82% proxy reduction and 97.64% edge strength. This is an engineering follow-up, not a fresh untouched primary test.

## Latency and GPU memory

| Measured mode | Time / throughput | Peak allocated | Peak reserved |
| --- | --- | ---: | ---: |
| Original full pipeline | Median 2.77 FPS; trials 2.61-2.92 | 385.69 MiB | 502 MiB |
| Efficient full pipeline | Median 4.34 FPS; trials 4.16-4.40 | 385.69 MiB | 502 MiB |
| UniDepthV2-S image inference | Median 58.37 ms/image | 485.84 MiB | See raw run record |
| VDA-S offline, input 518, 70 frames | 73.49 s total | 8014 MiB | 11858 MiB |
| VDA-S offline, input 392, 70 frames | 6.94 s total | 3072 MiB | 4194 MiB |

The two full-pipeline modes recompute depth, both flow directions, and fusion, with two warmups and three synchronized trials. Decode/resize and video encoding are excluded. Flow uses the CPU. Efficient throughput improves by 56.9%; these measurements do not establish real-time processing. Windows logical CUDA counters are not proof of physical VRAM residency. The main VDA reference uses the allowed input size 392. [Original trials](../results/runtime.json) and [efficient trials](../results/runtime-efficient.json) preserve stage timing and scope.

## Visual evidence

- [Pipeline](../results/figures/pipeline.png), [ablation tradeoffs](../results/figures/ablations.png), and [same-frame comparison](../results/figures/video-comparison.png).
- [Actual flow, warped recurrent history, support, and effective weight](../results/figures/flow-diagnostics.png): development frames 34-35, frozen primary configuration, reconstructed causal state from frame 0. Arrows use actual pixel displacement without magnification. A white support mask means accepted correspondence, not true depth.
- [Boundary crop](../results/figures/boundary-zooms.png): fixed primary frame 35, manually chosen geometry crop; not an automatic performance selection.
- [NYU failures](../results/figures/nyu-failures.png): the three highest per-image test AbsRel values, IDs 468, 87, 516, with [selection records](../results/failure-selection.json).
- [Stereo examples](../results/figures/stereo-comparison.png): Adirondack and Jadeplant, manually selected to contrast texture and thin geometry.

Local deliverables are the [seven-page report plus references](../report/independent-report.pdf), comparison and background-blur videos in runs/final/, faster exports in runs/efficient/, and outputs/depth_results_preview.zip. Videos/raw predictions/ZIP stay outside Git; the report, source, small measured summaries, and figures are tracked.

## Submission limit and final review

The linked official source decodes to 70 genuine frames at 24 FPS. All output videos retain those frames, their order, and rate. The requested 100-frame submission is not ready. Strict packaging rejects it; explicit provisional packaging includes a warning and submission_ready=false. Obtain the longer course source or TA clarification before final submission. The draft author line Eugene also needs final group confirmation.

The final code review found two important issues: short-limit calls could falsely claim course compliance, and archives could accept mismatched evidence. Both were reproduced and fixed with 10 regression tests; the complete suite passes 77 tests. One minor guard remains deferred: automatic enforcement that upstream checkouts are clean and match manifest revisions. The actual checked-out upstream HEADs match. Optional RAFT and affine alignment were not needed for this study. No main branch or groupmate implementation was modified.
