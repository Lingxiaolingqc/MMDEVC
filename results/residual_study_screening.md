# Controlled Residual screening: measured interpretation

All 22 configurations use seed 320112, 577 fitting images and 218 scene-separated validation images. This is development selection, not final test evidence. Three-seed confirmation is reported separately in residual_seed_confirmation.md. Each factor group uses the common R0, not a succession of winners.

## What the controls show

1. **Much of the simple improvement is calibration.** Frozen B0 AbsRel is 0.120846; a fitting-only scalar (B1) gives 0.112617. R0 gives 0.113218. R0 minus B1 is +0.000602, with scene-cluster bootstrap interval [-0.001791,+0.003064]. This does not establish that R0 adds accuracy beyond global scale correction.

2. **Frozen internal features are the strongest accuracy candidate.** AbsRel 0.100650 versus R0 0.113218; difference -0.012568, interval [-0.019705,-0.005382]. This is a single-seed, exploratory validation interval. F1 increases only from 0.460724 to 0.462552; its difference interval [-0.000912,+0.004975] crosses zero. Better metric depth does not establish sharper boundaries. The feature model has 28689 parameters versus R0 19969; a same-architecture feature-removal control is still needed before attributing all gain to information rather than added structure.

3. **Explicit edge inputs mainly affect the boundary diagnostic.** Joint Sobel features reach F1 0.465260, while AbsRel 0.113184 is effectively unchanged from R0 (difference -0.000034, interval [-0.000559,+0.000462]). The F1 difference interval [0.002949,0.006289] is positive in this initial seed, but multiple screening comparisons and seed uncertainty remain. These deterministic transforms add no independent sensor/GT information. They are not confirmed primary-accuracy improvements.

4. **The architecture/output tradeoffs differ.** Small U-Net: AbsRel 0.110817, F1 0.463323, 30681 parameters. Its AbsRel difference interval versus R0 crosses zero, so confirmation matters. Bounded log output: AbsRel 0.111149 but F1 0.453584; it does not dominate the additive reference. The MLP does not improve primary accuracy here. This common budget does not prove that an architecture or output family is intrinsically superior.

5. **The fixed auxiliary losses do little in this configuration.** Gradient loss and boundary weighting remain near R0 on AbsRel and F1. This is a negative result for the specified weights, optimization and architecture; it is not proof that all boundary-aware training is ineffective. Early checkpoint selection and limited spatial refinement are plausible explanations to examine, not established causes.

6. **The historical bundle improves accuracy with a boundary/cost penalty.** AbsRel 0.107814 and boundary-band AbsRel 0.164996 improve on its B3 base; F1 0.425977 is well below B0/R0. B3 already has F1 0.420877, so the entire boundary reduction cannot be blamed on its network. It also uses 388001 parameters and a second foundation-model flip pass. Several factors change together; no component-level causal claim follows from this row.

## Complete matched matrix

Positive delta means larger than R0: worse for AbsRel, better for F1. Metrics are image means; pooled boundary F1 is recorded separately in the machine-readable results.

### Base and preprocessing

| Method | AbsRel | Delta vs R0 | RMSE m | Delta1 | Boundary F1 | Delta F1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.113218 | +0.000000 | 0.469538 | 0.872506 | 0.460724 | +0.000000 | 19969 | 2 |
| base_scale | 0.112182 | -0.001036 | 0.465848 | 0.873772 | 0.454552 | -0.006173 | 19969 | 28 |
| base_flip | 0.113525 | +0.000307 | 0.469422 | 0.873837 | 0.431670 | -0.029054 | 19969 | 2 |
| base_flip_scale | 0.112373 | -0.000845 | 0.465335 | 0.875377 | 0.425180 | -0.035544 | 19969 | 28 |

### Inputs

| Method | AbsRel | Delta vs R0 | RMSE m | Delta1 | Boundary F1 | Delta F1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| depth_only | 0.111006 | -0.002212 | 0.459843 | 0.874272 | 0.461499 | +0.000775 | 19105 | 74 |
| flip | 0.113197 | -0.000021 | 0.469567 | 0.872506 | 0.460625 | -0.000099 | 20257 | 2 |
| confidence | 0.113200 | -0.000018 | 0.470375 | 0.872727 | 0.462357 | +0.001632 | 20257 | 22 |
| coordinates | 0.113157 | -0.000061 | 0.469757 | 0.873963 | 0.461649 | +0.000925 | 20545 | 8 |
| frozen_features | 0.100650 | -0.012568 | 0.441432 | 0.917826 | 0.462552 | +0.001828 | 28689 | 76 |
| rgb_sobel | 0.113214 | -0.000004 | 0.469737 | 0.872398 | 0.461567 | +0.000842 | 20257 | 2 |
| rgb_canny | 0.113190 | -0.000029 | 0.469648 | 0.872556 | 0.460895 | +0.000171 | 20257 | 2 |
| depth_sobel | 0.113127 | -0.000091 | 0.469182 | 0.872216 | 0.463784 | +0.003059 | 20257 | 2 |
| depth_canny | 0.113177 | -0.000041 | 0.469586 | 0.872579 | 0.460971 | +0.000247 | 20257 | 2 |
| flip_disagreement | 0.113176 | -0.000042 | 0.469579 | 0.872534 | 0.460684 | -0.000040 | 20545 | 2 |
| joint_edges | 0.113184 | -0.000034 | 0.469752 | 0.871568 | 0.465260 | +0.004536 | 20545 | 4 |

### Architecture

| Method | AbsRel | Delta vs R0 | RMSE m | Delta1 | Boundary F1 | Delta F1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | 0.113407 | +0.000189 | 0.470051 | 0.871704 | 0.460488 | -0.000237 | 19201 | 26 |
| unet | 0.110817 | -0.002402 | 0.462604 | 0.880629 | 0.463323 | +0.002599 | 30681 | 32 |

### Output representation

| Method | AbsRel | Delta vs R0 | RMSE m | Delta1 | Boundary F1 | Delta F1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| log_output | 0.112747 | -0.000471 | 0.467406 | 0.872015 | 0.453308 | -0.007416 | 19969 | 22 |
| bounded_log_output | 0.111149 | -0.002069 | 0.457374 | 0.875093 | 0.453584 | -0.007140 | 19969 | 52 |

### Loss

| Method | AbsRel | Delta vs R0 | RMSE m | Delta1 | Boundary F1 | Delta F1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient_loss | 0.113218 | +0.000000 | 0.469534 | 0.872506 | 0.460737 | +0.000013 | 19969 | 2 |
| boundary_loss | 0.113185 | -0.000033 | 0.469573 | 0.872575 | 0.460776 | +0.000052 | 19969 | 2 |

### Historical bundle

| Method | AbsRel | Delta vs R0 | RMSE m | Delta1 | Boundary F1 | Delta F1 | Parameters | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_bundle | 0.107814 | -0.005404 | 0.452478 | 0.884203 | 0.425977 | -0.034747 | 388001 | 20 |

## Qualitative evidence and limits

The first/middle/last validation images were fixed before looking at candidate scores. All three saved previews have been visually inspected for R0, frozen_features, unet and legacy_bundle; their local visual_review.md files retain the observations. For frozen_features, sample 0005 improves while 1440 worsens (AbsRel 0.088179 -> 0.134556). For U-Net, sample 0598 improves slightly (0.048129 -> 0.045762) but 1440 worsens (0.088179 -> 0.128646). Texture in a correction map is not proof of a depth boundary. Overview images do not establish new thin-structure recovery. Targeted success/failure crops remain for the final review. Other screening previews are retained but are not claimed to have all been visually reviewed.

## Next-stage decision

Before seeds 320113/320114, configs/residual_confirmation.json fixed R0 plus frozen_features, unet and legacy_bundle. The shortlist addresses primary accuracy, a pure architecture change, and the historical multi-factor tradeoff. Joint-edge findings remain a boundary-specific exploratory observation. No new test results were used. Subsequent combinations/removal controls, fixed-compute label-size curves and resolution controls are complete in their dedicated reports. Full795/654 final evaluation follows the frozen proposal.

Machine-readable evidence: outputs/part3_improvement/residual_study_20261118/analysis/{comparison.csv,matched_controls.csv,review.json}; per-run selected weights, source snapshots, complete history and per-image scores are retained. Bootstrap intervals resample scenes and preserve equal per-image weighting; they are not multiple-comparison corrected and do not measure training-seed uncertainty.

## Additional fixed-case inspection

joint_edges: The correction has narrow negative bands around existing table/chair contours in0005 and chair/counter contours in1440. The output still inherits the base silhouette; these overviews do not show recovery of the thin chair slats. A contour-shaped correction is not by itself evidence of better geometry.

bounded_log_output: In1440 the correction varies with broad regions and some appearance contours, including faint framed-picture shapes. The displayed output retains the base chair/counter layout. Appearance-correlated correction alone does not establish a true depth discontinuity.

gradient_loss: In0005 the correction is a broadly smooth negative field, with no obvious newly recovered fine geometry in the displayed refined depth.

boundary_loss: In1440 the correction is a broadly smooth negative field. The base chair/counter silhouettes remain; the overview does not establish sharper or newly recovered thin structures.

mlp: In1440 the correction is largely a smooth negative field with faint appearance variation; the displayed depth inherits the base structure.

Inspected image IDs and their exact per-image metrics are retained in each listed run's visual_review.md. This targeted inspection covers factor groups without claiming all previews or all218 validation images have been reviewed.
