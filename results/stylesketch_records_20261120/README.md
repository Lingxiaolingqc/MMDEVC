# SS1 complete StyleSketch-inspired Residual candidate: validation record

**Decision: retain F3 for the subsequent Residual study.** SS1 completes the requested
single-candidate experiment but does not beat F3 under the predeclared validation
AbsRel rule. Its boundary F1 improves, at higher cost; boundary-band depth error and
overall depth metrics do not improve on average. No sample-efficiency, LoRA search,
all795 fitting, or new654 test evaluation was run.

## Provenance and structure

Reference: [StyleSketch at 0214b46d51a7cd6e50674b4e4461adb3f720aedc](https://github.com/kwanyun/StyleSketch/tree/0214b46d51a7cd6e50674b4e4461adb3f720aedc).
The exact official `sketch/stylesketch_utils/stylesketch.py` and `conv.py` are archived
in `numerical_source.zip`, with verified Git blob IDs. This is an attributed adaptation
of its modules and topology, not a full reproduction or an original attention design.
UniDepthV2 Small is frozen at commit `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`.

| Method | Feature fusion before the identical common residual CNN |
|---|---|
|F3|Three genuine init/mid/late feature branches: independent 1x1 projection, resize to common grid, concatenate once.|
|P3|The same three branches fused coarse to fine with bilinear resize and a convolution/GELU at each step.|
|G3|P3 plus simplified spatial gating generated only from the accumulated feature.|
|SS1|A genuine initial feature and three native same-scale feature pairs; official-style two-input attention, resize plus learned convolution, BN/LeakyReLU, double-convolution intermediate fusion, final CBAM.|

All residuals combine the fused features with normalized RGB and log(D0)/2 in the
existing 32-channel, three-hidden-convolution CNN, predict additive deltaD, and restore
D=max(D0+deltaD,0.001). SS1 has 111145 trainable parameters vs F3's27089.

SS1 starts with init256x31x41 ->32. The fixed native pairs are:

| Level | Actual UniDepth outputs | Shape of each | Pair width |
|---|---|---|---|
|1|depth_layer.ups[0], process_features[1]|256x62x82|16+16|
|2|depth_layer.ups[1], process_features[2]|128x124x164|12+12|
|3|depth_layer.ups[2], depth_mlp[2]|64x248x328|8+8|

The final pair is related but distinct computation, including the real LayerNorm+Linear
depth feature. Pair inequality was verified for all795 images. There are no duplicated
tensors masquerading as independent levels.

Retained official operations: two-input Attention_block; nearest2x upsampling followed
by learned3x3 convolution (not transposed convolution); Conv+BN+LeakyReLU(0.2);
two convolutions at intermediate fusion levels; final concatenation directly into
CBAM with shared average/max channel attention and7x7 spatial attention. The four
adapted blocks matched official FP64 forward, input gradients and parameter gradients
exactly in both train and eval modes.

Task adaptations: three levels instead of eight, widths32/24/16, actual UniDepth feature
pairs instead of StyleGAN features, final CBAM32 ->1x1 projection16 ->240x320, RGB enters
the common depth CNN rather than the official final CBAM concat. No sketch output
Sigmoid, sketch training losses or sketch generator weights. Internal gating Sigmoids
are retained. Additional inputs/capacity mean this is a **complete-candidate comparison**,
not evidence that a fusion block alone caused any difference.

## Protocol and checks

577 fitting images /218 validation images, with disjoint187/62 scenes; seeds320112,
320113,320114. Course-clipped AbsRel loss, AdamW lr3e-4, weight decay0.001, batch8,
80-epoch cosine schedule to1e-5, gradient clip1, FP16 AMP, no augmentation; validation
every2epochs, minimum24epochs, stop after5 evaluations without strict improvement.
F3's68-epoch refit duration was not reused for SS1.

Inherited config fields for gradient/boundary loss weights and the earlier study's
zero-middle control are inactive here: the SS1 trainer calls the AbsRel loss only,
and this run executes no additional component control.

Depth metrics use y[45:471],x[41:601], clip[0.001,10]m, no test-GT scale alignment,
unweighted per-image means. Boundary edges: maximum neighboring log-depth jump>ln(1.1),
2px matching tolerance, eroded valid domain; boundary band2px. Table F1 is macro,
not pooled F1. Baseline B0 is a single deterministic frozen predictor, not three seeds.

`reference_audit.json` verifies split, protocol, training settings, cached inputs and
shared numerical source consistency before reusing F3/P3/G3/B0. The new cached init,
mid and late arrays also have identical SHA256 to the archived reference feature files.
The frozen foundation state was unchanged and all795 original depth predictions matched
exactly. Initial residual was exactly zero; fitting gradients reached every trainable
module after the initial zero-head warm-up. Validation BN buffers remained unchanged;
all17 BN update counters in best/final checkpoints equal only attempted fitting batches.
`training_audit.json` verifies all three runs, exact schedule, early stop, finite weights,
and no changes to old freezes or unrelated user edits.

Primary rule: smaller three-seed mean validation AbsRel; within0.2% relative of the
lowest, choose fewer parameters, then lower AbsRel. Only F3 and SS1 are eligible;
P3/G3/B0 are references. No post-hoc boundary-score veto or architecture search.

## Three formal SS1 runs

| Seed | Best/stop epoch | AbsRel | RMSE m | delta1 | Boundary F1 | Band AbsRel | Final AbsRel | Elapsed min | AMP skipped updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320112 | 22/32 | 0.095818 | 0.423911 | 0.922983 | 0.460363 | 0.157840 | 0.101095 | 86.47 | 2 |
| 320113 | 14/24 | 0.095718 | 0.431291 | 0.919813 | 0.465470 | 0.158862 | 0.098778 | 43.46 | 0 |
| 320114 | 14/24 | 0.095927 | 0.425102 | 0.922455 | 0.462129 | 0.158810 | 0.099452 | 43.01 | 0 |

Best weights define the primary metrics; final weights and per-image final metrics
are also retained. Seed320112 had two AMP skipped updates, both recorded; the other
seeds had none. Skipped optimizer steps do not mean missing fitting exposure or a
validation BN update. All six formal checkpoints are finite and hash-verified.

## Overall comparison

| Method | AbsRel down | RMSE m down | delta1 up | Boundary macro F1 up | Boundary-band AbsRel down |
| --- | --- | --- | --- | --- | --- |
| B0 UniDepth | 0.120846 | 0.475687 | 0.864816 | 0.451155 | 0.183047 |
| F3 | 0.094364 +/- 0.000835 | 0.413314 +/- 0.004322 | 0.925772 +/- 0.001233 | 0.452682 +/- 0.006282 | 0.154347 +/- 0.001468 |
| P3 | 0.094862 +/- 0.000734 | 0.416551 +/- 0.003988 | 0.925794 +/- 0.000559 | 0.455223 +/- 0.005794 | 0.155640 +/- 0.001579 |
| G3 | 0.095400 +/- 0.000287 | 0.420336 +/- 0.001388 | 0.925256 +/- 0.000913 | 0.461978 +/- 0.001931 | 0.156349 +/- 0.000809 |
| SS1 | 0.095821 +/- 0.000105 | 0.426768 +/- 0.003962 | 0.921750 +/- 0.001698 | 0.462654 +/- 0.002594 | 0.158504 +/- 0.000575 |

Values are mean +/- sample standard deviation across the three independently trained
models; this is not an ensemble. SS1 AbsRel is1.54% higher than F3 on average.

## Paired evidence

| Metric | SS1 minus F3 | 95% scene bootstrap interval |
| --- | --- | --- |
| abs_rel | +0.001457 | [-0.003071, +0.006361] |
| rmse_m | +0.013454 | [-0.007323, +0.038623] |
| delta1 | -0.004022 | [-0.013284, +0.004877] |
| boundary_f1 | +0.009972 | [+0.005771, +0.014107] |
| boundary_band_abs_rel | +0.004157 | [-0.000622, +0.008979] |

5000 paired scene-cluster bootstrap draws over62 scenes, RNG320118. Per-image errors
are averaged across three seeds before resampling; these confidence intervals do not
measure training-seed uncertainty. Validation was already used for earlier model
selection, so this remains exploratory reused-validation evidence without multiplicity
correction, not an unseen test result.

SS1 lowers AbsRel on106/218 images and raises it on112. Mean per-seed AbsRel differences
are +0.0019813, +0.0003914, +0.0019982. The overall AbsRel interval crosses zero: do not
claim a statistically established universal degradation. The boundary F1 interval is
positive. That evidence supports a modest boundary-detection gain, not lower depth
error near the same boundaries. The predeclared selection rule still chooses F3.

## Full training trajectories

The best epochs are22,14,14; final epochs32,24,24. After each best point, all five
subsequent validation checks fail to improve the minimum. Fitting loss falls from
approximately0.08981/0.09223/0.09099 at the best points to0.07993/0.08781/0.08094,
while final validation AbsRel rises to0.10110/0.09878/0.09945. Validation curves have
visible spikes, so the small standard deviation of selected best scores does not mean
smooth optimization. The pattern is consistent with overfitting/generalization-gap
growth. Online train-mode loss and eval-mode validation are not identical measurements;
without new component ablations we cannot assign the cause to BN, attention or a layer.
All epoch records and all12 old/new learning-curve figures are retained.

## Costs

| Method | Trainable parameters | Cached refiner median ms/image | Training allocated peak MiB | Recorded training min, seeds112/113/114 |
| --- | --- | --- | --- | --- |
| F3 | 27089 | 1.625 | 793.69 | 46.82, 69.82, 47.12 |
| P3 | 34049 | 2.062 | 861.07 | 53.25, 36.86, 42.27 |
| G3 | 34083 | 2.131 | 894.72 | 34.51, 51.47, 29.09 |
| SS1 | 111145 | 6.165 | 1684.74 | 86.47, 43.46, 43.01 |

Cached-refiner timing: RTX4060Laptop, FP16, batch8,10warmups and30measured repeated
fixed-image batches, forward plus output restore; excludes foundation, transfer andIO.
All four small refiner weights are resident during this benchmark. SS1 refiner latency
is about3.79x F3. Allocated-memory peaks are not total device memory or reserved-memory
requirements; training uses cached features and excludes a resident foundation model.

An additional same-session batch1 end-to-end benchmark over six fixed images
(6warmups,18measurements) includes UniDepth, observable-input preparation, transfers,
refiner and CPU depth output, but excludes disk/model loading. Median latency:
B0 31.688ms, F3 35.360ms, SS1 47.354ms. Corresponding allocated peaks:
351.980,352.087,352.453MiB. SS1 is about1.34x F3 end to end, not3.79x.
This short local measurement is not a universal performance benchmark.

Feature extraction of795 images took261.16s including cache writes, excluding final
hashing. The native cache is27.46GiB, plus3.18GiB storage-only input cache. These caches
and all weights remain local; publication contains their provenance and hashes.

Seed320112 wall time includes a2111s interval coinciding with severe host-memory
pressure (about121.6MiB free when inspected). We stopped only this task's verified
Python process after a complete epoch4 checkpoint and resumed it with a disk-backed
loader. All795 stored inputs were bitwise equal, the same checkpoint recomputed the
identical validation AbsRel0.10634412436827172, and optimizer/RNG/schedule resumed.
No model, loss, batch order or candidate width was changed. A separate frozen execution
supplement preserves the storage-only code and recovery checkpoints. The long interval
cannot be treated as pure GPU compute or conclusively attributed only to RAM pressure.
Training wall-time comparisons also have different early-stop durations.

## Qualitative inspection

The six fixed cases were declared before training; largest gain/regression are explicitly
post-hoc three-seed-mean AbsRel choices. Figures use seed320112, shared metric scales,
and an explicit evaluation mask. They do not use per-image normalization. Seven unique
images were manually inspected, not all218. The largest regression is also fixed0226.

| Sample | Role | F3 -> SS1 AbsRel | F3 -> SS1 F1 | Observed tradeoff |
| --- | --- | --- | --- | --- |
| nyuv2_0005 | fixed_0005 | 0.05244 -> 0.05264 | 0.36571 -> 0.38224 | Large room structure and chair/table contours are very similar. Some curved chair outlines look more continuous in SS1, but bright errors along boundaries and missed background/whiteboard depth edges remain. AbsRel is effectively unchanged/slightly worse; the F1 increase is small. |
| nyuv2_0598 | fixed_0598 | 0.04018 -> 0.04706 | 0.49453 -> 0.54176 | SS1 makes part of the table outline more continuous in the thresholded edge map. Thin legs and sofa contours remain incomplete. Errors over the left foreground sofa become larger. This is a clear example of better edge detection F1 without better overall depth accuracy. |
| nyuv2_1440 | fixed_1440 | 0.13105 -> 0.10160 | 0.47280 -> 0.51322 | SS1 reduces broad background/room depth error and changes some chair gaps/contours. Large errors on foreground chairs and the right countertop persist; neither model recovers all GT boundaries. Both AbsRel and F1 improve for this seed/image. |
| nyuv2_0226 | fixed_0226 | 0.21835 -> 0.41701 | 0.51423 -> 0.51172 | SS1 pushes a large background wall/pillar/furniture region too far away, producing broad high errors. F3 is substantially closer to GT in those areas, while their gross outlines remain similar. Fine branches remain inaccurate in both. This is also the post-hoc largest three-seed-mean AbsRel regression; it is not an additional independent example. |
| nyuv2_0246 | fixed_0246 | 0.14330 -> 0.09211 | 0.51237 -> 0.51060 | SS1 increases background depth and reduces F3 underestimation over much of the wall/room. Some foreground/lamp/ceiling errors remain. Depth accuracy improves while F1 is nearly unchanged/slightly lower: the gain is not simply sharper boundaries. |
| nyuv2_0080 | fixed_0080 | 0.08564 -> 0.11008 | 0.09360 -> 0.35273 | Both models miss most of the GT thin hanging-object boundary under the fixed edge threshold. SS1 recovers a few more line fragments, giving a large F1 increase from a very low base, but broad curtain/wall depth errors increase. The edge plot does not support a claim of complete thin-structure recovery. |
| nyuv2_0244 | largest_gain | 0.13281 -> 0.07747 | 0.54998 -> 0.56891 | SS1 corrects substantial underestimation across the background wall and much of the room. The left bright opening is overestimated and floor errors remain. Only a few additional depth-edge fragments are present. This is a disclosed post-hoc favorable case, not representative evidence alone. |

The contrasting0226 and0246 cases show that a broad depth correction can help one scene
and damage another. A more convincing contour alone cannot validate metric accuracy.
Full observations, exact inspected-figure hashes and limits are in
`analysis/visual_review.json`; per-case recomputed metrics are in `analysis/visuals/metadata.json`.

## Reproduction and retained evidence

Main entry points in the repository:

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/part3_improvement/run_stylesketch_residual.py prepare
& $py -X utf8 scripts/part3_improvement/run_stylesketch_residual.py preflight
& $py -X utf8 scripts/part3_improvement/run_stylesketch_residual.py train
& $py -X utf8 scripts/part3_improvement/analyze_stylesketch_residual.py --visuals
```

These commands depend on the documented local NYUv2 data, original checkpoints and
prior cached-reference records. Preserve existing completed outputs; for a fresh
reproduction use a separate checkout/output configuration and freeze it before training.
The normal loader needs sufficient host RAM. The actual run's
`run_stylesketch_low_memory.py check/train` is an explicit recovery path that expects
the saved recovery checkpoint; it is not advertised as a fresh-run initializer.
Recorded frozen numerical sources must be byte-identical, including line endings.

Publication contains full configs, reference audits, source freezes/ZIPs, curves,
per-image best/final scores, logs, failure and recovery records, and selection evidence.
Six formal best/final weights plus two engineering recovery states remain under
`outputs/part3_improvement/stylesketch_20261120/`; `local_weights.json` records exact
paths and hashes. Native features and inputs are local and reproducible from their
source/hash manifests. Local publication/audit helpers stay in ignored outputs, not
the teacher-facing scripts directory. No previous experiment or freeze was overwritten.

Final choice: **F3**. SS1 is a completed, informative alternative with a boundary F1
benefit, higher cost, and no demonstrated overall accuracy advantage. The next
sample-efficiency or LoRA comparison is not started by this experiment.
