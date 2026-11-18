# Controlled residual refinement of frozen UniDepthV2

**Status:** development experiments, frozen all-development refits and complete official test evaluation are finished. This section contains measured results and retained limitations.

## Research question and protocol

We investigate whether a small supervised residual network can improve the metric-depth accuracy of the course's frozen UniDepthV2 Small model, and whether any improvement also recovers depth boundaries. This is a separate metric-image study; it does not combine LoRA with the earlier DA video-fusion method.

The official NYUv2 training set is divided into 577 fitting images and 218 validation images with disjoint scenes (187 and 62 scenes). Input, architecture, output and loss choices use this development split. The 654 official test images are excluded from the new design-selection process. Earlier project work already inspected test results, so the eventual frozen evaluation must not be described as a never-seen blind test.

Evaluation uses the original 480×640 images, crop `[45:471, 41:601]`, valid ground-truth depths in `[0.001, 10]` metres and predictions clipped to the same range. We use raw metre-valued NYUv2 ground truth and average per-image AbsRel, RMSE and strict δ1 scores equally. No per-image ground-truth scale fitting is permitted. Boundary diagnosis uses a 10% neighboring log-depth jump, 2-pixel matching tolerance and a fixed eroded evaluation domain. Boundary-band AbsRel measures metric error within two pixels of GT discontinuities; it answers a different question from boundary F1.

## Residual model

Let $D_0$ denote the frozen prediction, $I$ the RGB image, and $F$ the frozen model's native `depth_features` tensor (256×31×41 in this inference configuration). At the initial 240×320 refinement resolution, the input is

\[
X=\operatorname{concat}\left[A(I)/127.5-1,\;
\tfrac12\log\max(A(D_0),10^{-3}),\;
B(P_{1\times1}(F))\right].
\]

Here $A$ is area resizing, $P_{1\times1}$ is a learned 256-to-16 channel projection, and $B$ is bilinear resizing. Three 3×3 convolution/GELU layers with width 32 and a final 3×3 single-channel layer predict a residual in metres. The final layer is initialized to zero, preserving the frozen prediction at initialization. The restored prediction is

\[
\widehat D=\max\left(D_0+B_{480\times640}(R_\theta(X)),10^{-3}\right).
\]

The refiner has 28,689 trainable parameters; the foundation model remains frozen. Training minimizes the course-clipped, valid-GT, per-image AbsRel objective. The basic reference R0 uses the same CNN with only RGB and predicted log depth (19,969 parameters). Inference features never include GT depth, GT edges or evaluation masks.

## Controlled development experiments

We first compared the frozen model (B0), one global fitting-only scale (B1), a fixed 25% restored flip-prediction blend (B2), and that blend plus a fitting-only scale (B3). Validation AbsRel was 0.120846, 0.112617, 0.121375 and 0.112904 respectively. The fixed flip average did not improve this baseline; scalar calibration is an essential reference for judging a learned correction.

Twenty-two single-seed configurations independently varied the base/input, architecture, output or loss relative to R0. Inputs included the historical flip prediction, confidence, coordinates and internal features, RGB/depth Sobel or Canny maps, signed/absolute flip disagreement, and joint RGB/depth gradients. Architecture controls used a pixelwise MLP, the CNN and a small U-Net. Output controls used additive, log-multiplicative and bounded log-multiplicative residuals. Loss controls added a fixed log-gradient term or GT-boundary weighting. The historical multi-factor bundle was retained as an overall reference, not evidence for its individual components.

All screening runs used seed 320112, AdamW, learning rate 3×10⁻⁴, batch size 8, an 80-epoch cap, the same cosine schedule and validation/early-stop rules. R0 and three selected candidates were then evaluated with seeds 320112/320113/320114. Adding internal features also changes the learned input projection and parameter count. The equal-parameter zero-feature control below tests information in C1; it does not independently isolate capacity in the final CNN. The complete matrix, including unchanged and negative findings, remains in [the screening analysis](residual_study_screening.md).

| Method | Validation AbsRel, mean ± SD | Boundary macro F1, mean |
|---|---:|---:|
| R0 | 0.113272 ± 0.000143 | 0.460822 |
| Internal-feature CNN | 0.099848 ± 0.000875 | 0.462688 |
| Small U-Net without internal features | 0.110363 ± 0.000401 | 0.463782 |
| Historical bundle | 0.107104 ± 0.000621 | 0.423720 |
| Internal-feature U-Net (C1) | 0.099889 ± 0.000796 | 0.461370 |

For the internal-feature CNN versus R0, the mean per-image AbsRel change across the three seeds was −0.013424, with a scene-cluster bootstrap 95% interval of [−0.021064, −0.005393]. Boundary-band AbsRel also decreased, but the F1 change of +0.001866 had an interval crossing zero. These intervals resample validation scenes; training-seed SD is reported separately. Averaging per-image scores is not prediction ensembling, and these exploratory intervals are not corrected for multiple comparisons.

Combining internal features with the small U-Net did not establish a further gain. C1 minus the internal-feature CNN changed AbsRel by +0.000041, interval [−0.003525, +0.003563], with slightly lower mean F1. A same-parameter zero-feature control worsened AbsRel to 0.109718 in seed 320112, supporting the usefulness of internal information in that structure. Removing RGB slightly reduced single-seed AbsRel but lowered F1, failing the predeclared promotion rule. We therefore retained the simpler internal-feature CNN for subsequent controls. See [the combination analysis](residual_combinations.md) for the control's effective-capacity caveat and visual evidence.

## Label size, training duration and visual limits

Eight further runs compared R0 and the internal-feature CNN on nested whole-scene subsets using 2,000 attempted updates and exactly 16,000 image exposures. Every run had zero recorded AMP skipped steps. The main table uses the final iterate; earlier validation-selected weights remain separate.

| Fitting images | Scale-only AbsRel | R0 AbsRel | Internal-feature CNN AbsRel |
|---:|---:|---:|---:|
| 58 | 0.113809 | 0.115481 | 0.109264 |
| 145 | 0.112730 | 0.116284 | 0.108468 |
| 289 | 0.112756 | 0.114445 | 0.104614 |
| 577 | 0.112617 | 0.113489 | 0.103307 |

The internal-feature model's point estimate was lower at each size, although the paired interval at 58 labels crossed zero. Boundary F1 did not improve over the corresponding R0. The 58-label validation curve reached 0.106389 at 400 updates and then worsened while training loss continued to fall, consistent with overfitting under this exposure budget. The 577-label validation curve largely improved across training. This is one seed and one nested scene ordering, not a universal data-efficiency law. The differing cosine horizons and checkpoint-selection rules prevent treating the screening-versus-fixed-budget gap as an isolated duration effect.

Fixed first/middle/last validation cases reveal persistent limitations. For image 1440, the frozen model's AbsRel was 0.088179, whereas all three confirmed internal-feature CNNs worsened it to approximately 0.131–0.135. U-Net corrections looked smoother but did not remove this failure. In the fixed-budget study, increasing labels from 58 to 577 reduced this image's error from 0.155789 to 0.123666, still worse than B0. Overview contours mostly remained inherited from UniDepth; they do not establish recovered thin structures. An additive correction can also cross the boundary detector's relative-jump threshold without creating a new contour, so F1 must be interpreted alongside metric error and images. Full curves and inspection scope are recorded in [the data-size analysis](residual_data_size.md) and [the boundary interpretation](residual_boundary_interpretation.md).

## Resolution and final selection

With the same seed and training protocol, raising the internal-feature refiner from 240x320 to 480x640 changed AbsRel from 0.100650 to 0.100712 and boundary F1 from 0.462552 to 0.463258. Both paired intervals crossed zero, and neither predeclared promotion requirement was met. Training peak allocated memory rose from 436.34 to 1579.96 MiB. Fixed-case overviews remained very similar; a more textured correction map did not establish recovered contours. We therefore retained 240. See [resolution sensitivity](residual_resolution.md).

Before any new test inference, we froze R0 and the internal-feature CNN with three seeds each. All 795-development refitting uses 2 epochs for R0 and 76 epochs for the internal-feature CNN: the median selected epochs across development seeds. The cosine horizon remains 80 epochs. Final methods are evaluated against B0–B3, with global scales fitted only on the 795 development images. Each method uses its fixed final iterate; no test checkpoint selection is permitted. Different final training durations are development-selected and are not a matched-compute comparison.

## Runtime and final evaluation

An isolated local RTX 4060 Laptop measurement using development-selected checkpoints of these architectures, with decoded RGB input and float32 CPU depth output, averaged 24.758 ms for B0, 28.899 ms for R0, 30.287 ms for the internal-feature CNN, 32.706 ms for C1 and 67.739 ms for the historical flip bundle. This includes required foundation/flip inference, transfers, input formatting and refinement, but excludes file IO and visualization. The selected CNN adds about 22.3% latency in this short benchmark. See [runtime scope and measurements](residual_runtime.md); these are not universal hardware-speed claims.

## Frozen final results

| Method | Test AbsRel | RMSE m | delta1 | Boundary macro F1 |
|---|---:|---:|---:|---:|
| Frozen UniDepthV2 (B0) | 0.110776 | 0.392817 | 0.894781 | 0.456852 |
| Global scale (B1) | 0.102179 | 0.393628 | 0.896240 | 0.456898 |
| Flip blend (B2) | 0.110101 | 0.390022 | 0.896409 | 0.423111 |
| Scale + flip (B3) | 0.101503 | 0.391522 | 0.897630 | 0.423129 |
| R0, three-seed mean | 0.101263 | 0.388395 | 0.900296 | 0.468205 |
| Internal-feature CNN, three-seed mean | **0.090448** | **0.362062** | **0.932393** | **0.469133** |

The refiner lowers AbsRel by 18.35% relative to B0. Its seed SD is 0.000034; R0 SD is 0.000468. The paired AbsRel change versus B0 is-0.020328, with a 215-scene bootstrap 95% interval[-0.024628,-0.016295]. The change remains negative versus B1[-0.015525,-0.008169], B3[-0.014917,-0.007506] and R0[-0.014449,-0.007465]. This is supervised NYUv2 adaptation under the fixed course protocol, not a zero-shot or universal SOTA claim. B0 exactly reproduces the earlier saved course evaluation.

Boundary F1 improves over B0, but the additional change over R0 is only+0.000928, interval[-0.001166,+0.002914]. Boundary-band AbsRel nevertheless decreases from R0's 0.169446 to 0.160558. This supports more accurate depths near boundaries, without establishing further contour recovery beyond the basic refiner.

Improvements are heterogeneous: 433 of 654 images have lower mean-seed AbsRel than B0, while 221 worsen; versus R0 the count is 357 improved and 297 worsened. The median change versus B0 is-0.019563, and the ten largest gains contribute 8.62% of gross improvement. The gain is not solely a few extreme examples, but is not a per-image guarantee.

The fixed/median/ranked panels preserve both benefits and failures. Image 1207 improves from 0.254005 to 0.084861 in the predeclared representative seed, mainly through broad sofa/floor depth correction; fine background structures remain missing. Image 0088 improves substantially yet retains a grossly wrong central region. Images 0770,0327 and0086 regress;0086 worsens from 0.089054 to 0.261981 over nearby shelves/floor. Images 0086 and0088 belong to the same named bookstore scene, so even neighboring views need not benefit equally. These statements come from inspected depth crops and error-change maps, not appearance plausibility alone.

All six final models use fixed final iterates. The three internal-feature runs each recorded two AMP nonfinite-gradient events, retained in their logs; R0 recorded none. No test image was omitted. Full per-seed results, paired intervals, selection rules and eight inspected cases appear in [the final test analysis](residual_final_test.md). No test-driven model revision followed this evaluation.
