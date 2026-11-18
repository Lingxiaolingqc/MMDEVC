# Residual label-size and training-curve comparison

Status: all eight fixed-compute jobs completed on the fixed218-image validation set; no new test inference.

Protocol: nested whole-scene subsets58/145/289/577 images (18/47/95/187 scenes), ordering seed320118, training seed320112. R0 and the selected internal-featureCNN use2000 attempted updates, batch8, exactly16000 image exposures, the same2000-step cosine schedule and no early stopping. All eight runs recorded zero AMP skipped steps. Main scores use the final iterate; best-validation checkpoints are retained separately.

| Fitting images | Scenes | B0 AbsRel | Subset-scale B1 | R0 AbsRel | Internal-featureCNN AbsRel | R0 F1 | Internal-featureCNN F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 58 | 18 | 0.120846 | 0.113809 | 0.115481 | 0.109264 | 0.465200 | 0.463423 |
| 145 | 47 | 0.120846 | 0.112730 | 0.116284 | 0.108468 | 0.462624 | 0.459819 |
| 289 | 95 | 0.120846 | 0.112756 | 0.114445 | 0.104614 | 0.461170 | 0.459914 |
| 577 | 187 | 0.120846 | 0.112617 | 0.113489 | 0.103307 | 0.461656 | 0.460509 |

The internal-feature model has lower final AbsRel than R0 at every tested size and is also lower than the subset-fitted scalar reference. R0 does not beat the scalar reference in these point estimates. Only the neural pair has matched optimizer/exposure budgets; scalar calibration is a same-label-set contextual baseline.

Internal-featureCNN minus R0 AbsRel differences and paired scene-bootstrap95% intervals:
- n=58: -0.006217 [-0.014468,+0.001695].
- n=145: -0.007816 [-0.015594,-0.001527].
- n=289: -0.009831 [-0.017980,-0.003350].
- n=577: -0.010183 [-0.018306,-0.003505].

The n58 interval includes zero. The other three AbsRel intervals are negative conditional on this training seed/subset ordering. These are exploratory, unadjusted intervals across validation scenes, not repeated-training uncertainty or a universal sample-complexity claim. BoundaryF1 is slightly lower than R0 at every size; each corresponding F1 interval includes zero. Band-depth error is lower, with negative intervals at145/289/577. Accuracy near an existing boundary and recovering a more complete contour are distinct outcomes.

## Training-time behavior

For internal features with58 labels, validation reaches0.106389 at400 updates and ends at0.109264, while the training objective decreases from0.099536 in the first200-update block to0.082189 in the last. The145-label run similarly reaches0.106240 at200 updates and ends at0.108468. This is consistent with overfitting under a fixed long exposure budget; the main table intentionally does not substitute the earlier best scores.

With577 labels, validation largely decreases across the common trajectory: the best is0.103293 at1800 updates and the final result is0.103307. The fixed-budget result must not be directly contrasted with the80-epoch screening result as an isolated duration effect because their cosine horizons and checkpoint-selection rules differ. Complete histories and both kinds of checkpoint remain available.

## Qualitative inspection

Inspected the comparison plot, the58/577 internal-feature training curves, and fixed0005/1440 previews at those two data sizes. The58-label corrections look substantially more patch-like and stronger than577 in these examples. The displayed depth contours remain largely inherited from UniDepth. This observation is limited to those fixed cases.

For0005, AbsRel improves from0.062944 (58 labels) to0.053101 (577); B0 is0.078315. For1440, error decreases from0.155789 to0.123666, but remains worse than B0 at0.088179. Its boundary-band error similarly remains worse:0.251608/0.227368 versus B0 0.189251. More data reduces this particular failure but does not remove it. No claim of thin-structure recovery is supported by these overview images.

Evidence: analysis/data_size/review.json and comparison.png; each size run retains per-image scores, full learning curves, selected-final/best weights, source snapshots and fixed previews. The earlier Windows snapshot failure was pre-training, retained, repaired and resumed with unchanged numerical settings.
