# How to interpret the Residual boundary results

This note concerns the implemented diagnostic, not an additional trained method. The code in src/p3_depth/evaluation/boundary.py marks a pixel if its maximum valid four-neighbor log-depth difference exceeds log(1.1), then matches predicted/GT boundary pixels within2px inside the eroded evaluation domain. MacroF1 and pooledF1 have different weighting and are reported separately.

A change in boundary F1 does not necessarily mean a new object contour was recovered. For example, neighboring depths2.00m and2.19m have ratio1.095, below the1.1 boundary threshold. Subtracting the same0.15m from both gives1.85m and2.04m, ratio1.1027, which crosses the threshold while keeping the boundary position and the absolute0.19m difference unchanged. Multiplicative global scaling preserves these ratios except for clipping/validity effects; additive correction does not.

Consequently, the broad negative corrections seen in R0 and several refiners can change boundary detection even when overview contours look almost unchanged. This is a possible contribution, not a measured attribution of the entire F1 change in our learned models. We therefore interpret boundary F1 alongside boundary-band metric error, full-image accuracy and fixed-case visualization. A higherF1 alone is not proof of thin-structure reconstruction; a lower band error does not guarantee more complete contours.

This does not invalidate the fixed metric: all compared methods use the same rule. It limits the wording of the conclusion, especially for small single-seed F1 differences and scale/offset-sensitive refinements.
