# Residual seed confirmation — development evidence

Updated 2026-11-18. All four methods now have the three declared seeds (30 completed formal screening/confirmation runs in total). No new final test evaluation.

| Method | Completed seeds | Mean AbsRel | Sample SD | Mean boundary F1 |
|---|---:|---:|---:|---:|
| R0 | 3/3 | 0.113272 | 0.000143 | 0.460822 |
| frozen_features | 3/3 | 0.099848 | 0.000875 | 0.462688 |
| unet | 3/3 | 0.110363 | 0.000401 | 0.463782 |
| legacy_bundle | 3/3 | 0.107104 | 0.000621 | 0.423720 |

For frozen_features minus R0, the mean per-image error across three independently trained seeds gives AbsRel change -0.013424 (scene-cluster 95% interval [-0.021064,-0.005393]), RMSE change -0.033397 m ([-0.055214,-0.009555]), and boundary-band AbsRel change -0.009165 ([-0.014092,-0.003576]). Boundary F1 change is only +0.001866 ([-0.000771,+0.004855]). These intervals resample validation scenes; seed SD is reported separately. Averaging error scores across seeds is not a prediction ensemble.

Gains are heterogeneous. For seeds 320112 and 320113, 136/218 and 124/218 images improve over the corresponding R0; 82 and 94 worsen. The median differences (-0.003701 and -0.003149) are smaller in magnitude than the mean differences (-0.012568 and -0.013455). The largest ten improvements account for 37.2% and 35.3% of gross error reduction (excluding losses in the denominator). Do not present the mean improvement as universal.

Fixed sample 1440 is a persistent failure: frozen baseline AbsRel 0.088179 versus internal-feature refiners 0.134556, 0.131380 and 0.130573. Boundary-band errors also worsen across all three seeds. The seed320114 preview shows patch-like corrections and mostly preserved base contours, without clearly recovered thin structures. A larger spatial-context refiner is a motivated hypothesis to test after confirmation finishes, not a demonstrated solution.

The third-seed fixed previews (0005/0598/1440) were visually inspected and recorded beside that run. Aggregate tables, paired scene intervals, benefit distributions and all per-image records remain in the study output.
