# Frozen final NYUv2 test evaluation

Status: complete. All six all 795-development refits and all ten 654-image evaluations finished. Methods, seeds, durations and numerical sources were frozen before the new test cache. Earlier project test exposure is disclosed; this is not a never-seen blind test.

## Main results

| Method | AbsRel | Seed SD | RMSE m | delta1 | Boundary macro F1 | Boundary-band AbsRel |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.11077566 | n/a | 0.392817 | 0.894781 | 0.456852 | 0.180059 |
| B1 | 0.10217932 | n/a | 0.393628 | 0.896240 | 0.456898 | 0.168251 |
| B2 | 0.11010056 | n/a | 0.390022 | 0.896409 | 0.423111 | 0.178586 |
| B3 | 0.10150252 | n/a | 0.391522 | 0.897630 | 0.423129 | 0.166718 |
| R0 | 0.10126258 | 0.00046836 | 0.388395 | 0.900296 | 0.468205 | 0.169446 |
| frozen_features | 0.09044777 | 0.00003434 | 0.362062 | 0.932393 | 0.469133 | 0.160558 |

B0: original UniDepthV2; B1: all-development global scale; B2: fixed 25% restored flip blend; B3: blend plus all-development global scale. Neural rows average per-image scores across three independent seeds, not predictions. Each final model has its own retained predictions and score table.

The internal-feature CNN lowers AbsRel by 18.35% relative to B0, lowers RMSE from 0.392817 to 0.362062 m, and raises delta1 from 0.894781 to 0.932393. Its supervised NYUv2 adaptation adds 28,689 trainable parameters to a frozen foundation model. This is a result under the recorded course protocol, not a universal SOTA or cross-dataset claim.

B1 reduces AbsRel but slightly increases RMSE. B2/B3 lose boundary F1 despite their depth-error tradeoffs. R0 captures much of the boundary-score increase; the internal-feature model does not establish an additional F1 gain over R0.

## Paired uncertainty and benefit distribution

| Reference | CNN minus reference AbsRel | Scene-bootstrap 95% interval | Images improved / worsened | Median change |
|---|---:|---|---:|---:|
| B0 | -0.02032789 | [-0.02462789, -0.01629500] | 433 / 221 | -0.01956262 |
| B1 | -0.01173154 | [-0.01552467, -0.00816891] | 382 / 272 | -0.00329979 |
| B2 | -0.01965278 | [-0.02396229, -0.01561662] | 434 / 220 | -0.01987680 |
| B3 | -0.01105474 | [-0.01491675, -0.00750602] | 373 / 281 | -0.00258228 |
| R0 | -0.01081481 | [-0.01444861, -0.00746517] | 357 / 297 | -0.00124125 |

Intervals use 5000 paired cluster draws over 215 test scenes, retaining image weighting. Training-seed SD is separate; these intervals do not jointly measure seed uncertainty and are exploratory without multiple-comparison correction. All methods use 654 images and 156018240 GT pixels. No invalid predictions/images were silently omitted.

Boundary F1 change versus B0 is+0.012280, interval[+0.009643,+0.015004]. Versus R0 it is+0.000928, interval[-0.001166,+0.002914]. Boundary-band AbsRel versus R0 decreases by 0.008889, interval[-0.011413,-0.006512]. Better depth values near edges are distinct from recovering the correct contour. Relative-jump threshold effects remain relevant; see residual_boundary_interpretation.md.

The ten largest positive contributions account for 8.62% of gross improvement versus B0 and 17.73% versus R0; the mean gain is not solely ten extreme examples. Nevertheless, 221 images worsen versus B0 and 297 versus R0. The median increment beyond R0 is only-0.001241; a strong average does not guarantee per-image improvement.

## Every frozen neural run

| Method / seed | Epochs | Attempted updates | AbsRel | RMSE m | delta1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| R0 / 320112 | 2 | 200 | 0.10097711 | 0.388223 | 0.900527 | 0.468398 |
| R0 / 320113 | 2 | 200 | 0.10180311 | 0.386875 | 0.900893 | 0.466102 |
| R0 / 320114 | 2 | 200 | 0.10100752 | 0.390087 | 0.899469 | 0.470115 |
| frozen_features / 320112 | 76 | 7600 | 0.09042590 | 0.362258 | 0.932508 | 0.469229 |
| frozen_features / 320113 | 76 | 7600 | 0.09043008 | 0.363120 | 0.932574 | 0.469882 |
| frozen_features / 320114 | 76 | 7600 | 0.09048735 | 0.360809 | 0.932097 | 0.468287 |

R0 used 2 epochs and 1590 image exposures; internal-feature CNN used 76 epochs and 60420 exposures. Each CNN refit recorded two AMP nonfinite-gradient events; R0 recorded none. All events and histories are retained. Counts above are attempted updates. Durations were selected by median development best epoch with the original 80-epoch cosine horizon, so this final comparison is not a matched-training-compute study. The earlier 2000-update study provides that control.

## Inspected qualitative cases

All eight selected overview/crop panels were visually inspected, plus error-change maps for 0088,1207,0327,0086 and0770. Displayed model seed 320112 was fixed before test. Ranking uses three-seed mean per-image AbsRel differences versus B0: two greatest improvements, two greatest regressions, median, and fixed first/middle/last. Crops use a fixed GT-edge-density rule only for diagnosis, never model input/selection.

| Image / scene | Selection | B0 AbsRel | Displayed CNN AbsRel | Observation |
|---|---|---:|---:|---|
| nyuv2_0001 / kitchen_0004 | fixed first | 0.093337 | 0.058432 | Broad depth values improve over B0, but the scalar B1 is better than the CNN on this fixed kitchen case. The close-up retains the base object silhouette; no new fine structure is established. |
| nyuv2_0770 / kitchen_0023 | fixed middle | 0.024301 | 0.068882 | A real fixed-case regression: the original prediction is already close to GT. The refiner shifts broad regions and adds spatially varying error across cabinets/counter areas. Its silhouette remains plausible despite worse metric accuracy. |
| nyuv2_1449 / dining_room_0036 | fixed last | 0.093533 | 0.077828 | Overall accuracy improves modestly. The chair/table opening in the crop remains shaped much like B0 and does not recover the detailed GT boundary. |
| nyuv2_0088 / bookstore_0001 | two largest mean-score improvements | 0.619818 | 0.316234 | Large foreground depth errors decrease over books/display surfaces. The central background/person region remains grossly wrong and saturated in the depth/error display; parts of the ceiling worsen. A large mean gain does not make this a solved scene. |
| nyuv2_1207 / living_room_0027 | two largest mean-score improvements | 0.254005 | 0.084861 | Broad errors over the sofa and floor decrease substantially, and background depth moves toward GT. Fine objects/curtain transitions behind the sofa remain missing; narrow sofa/plant boundary regions can worsen. |
| nyuv2_0327 / classroom_0023 | two largest mean-score regressions | 0.124555 | 0.249315 | Major regression: the refiner increases depth over broad classroom regions already overestimated by B0; the error-change map is predominantly worse across the room and foreground. Existing silhouettes remain plausible and cannot reveal accuracy alone. |
| nyuv2_0086 / bookstore_0001 | two largest mean-score regressions | 0.089054 | 0.261981 | Major regression: errors grow over the nearby shelves and floor while only limited far areas improve. This is the same named bookstore scene as successful image 0088, so success cannot be generalized even to every view of that scene. |
| nyuv2_0475 / study_0002 | median mean-score change | 0.075741 | 0.056591 | The median-change case improves broad depth values. Table-leg boundaries largely retain their base positions and blur; the crop does not demonstrate newly recovered fine geometry. |

The main visual effect is regional depth correction. Fine silhouettes often remain inherited from UniDepth; new thin-structure recovery is not established. The three genuine regressions 0770,0327 and0086 are retained explicitly. These are post-evaluation diagnoses, not grounds for a new test-driven revision.

## Evidence and reproduction

Study root: outputs/part3_improvement/residual_study_20261118/. Final results: final_test/; paired tables/PNG/PDF: analysis/test/; fixed and ranked images: analysis/test/cases/. Fresh B0 exactly reproduces prior saved course and boundary scores (baseline_reproduction.json). Every formal training run retains source/config/history/weights; record_retention_audit.json verifies all 51 plus one engineering smoke.

Run analyze_residual_followups.py --section test and render_refiner_cases.py to regenerate the analysis from retained predictions. Frozen model settings and final methods are unchanged; these scripts only analyze results. [Full reading/reproduction guide](../docs/residual_results_guide.md).
