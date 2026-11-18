# Residual combination study — validation only

All three C1 seeds and both removal controls are complete (five new jobs). No new test inference.

C1 combines the frozen native 256-channel decoder features (learned 1x1 projection to 16) with the small U-Net. RGB/log-depth inputs, original frozen depth base, additive output, AbsRel loss, split and training protocol remain unchanged. Exact trainable parameters: C1 35,945; internal-feature CNN 28,689; U-Net without internal features 30,681. The input projection is learned; the original UniDepthV2 is frozen.

| Method | Seeds | Mean AbsRel | SD | Mean boundary F1 |
|---|---:|---:|---:|---:|
| Internal-feature CNN | 3 | 0.099848 | 0.000875 | 0.462688 |
| C1 internal-feature U-Net | 3 | 0.099889 | 0.000796 | 0.461370 |
| U-Net without internal features | 3 | 0.110363 | 0.000401 | 0.463782 |

C1 minus the internal-feature CNN has mean AbsRel change +0.0000414, paired scene-bootstrap 95% interval [-0.003525,+0.003563]. Boundary F1 change is -0.0013185, interval [-0.002950,+0.000220]. Thus there is no established incremental benefit from replacing this CNN with this U-Net under the tested protocol. This does not prove that all U-Net designs are equivalent or worse.

C1 versus U-Net without internal features reduces mean AbsRel by 0.0104733 (interval [-0.016192,-0.005058]); boundary-band AbsRel falls by 0.0057561 ([-0.009016,-0.002630]), but boundary F1 does not improve. This supports information from the internal representation for metric correction, not a general claim of sharper edges.

The same-parameter zero-feature control (seed320112) gets AbsRel 0.109718 and F1 0.464441, versus C1's 0.099144 and 0.461957. It retains the 35,945 parameter architecture while feeding zero internal maps. The projection weights then receive no input-dependent gradient, so equal parameter count is not equal effective functional capacity. Treat this single-seed control as supporting evidence, not a three-seed confirmation.

Qualitative review: C1 seed320112's three fixed previews were inspected; sample1440 was also inspected for the other two seeds. Its correction is smoother than the original internal-feature CNN example, yet it still degrades this sample versus frozen UniDepth. The three-seed accuracy result is consistent with not claiming a structural gain. Detailed fixed-case reviews are retained beside each inspected run.

All intervals average per-image scores across seeds where stated; they are not prediction ensembles. Scene resampling and seed SD measure different uncertainties. Exploratory comparisons are not adjusted for multiple tests.

The no-RGB control gets AbsRel 0.098915 and boundary F1 0.460682, versus full C1 seed320112 at 0.099144 and 0.461957. It fails the predeclared non-decreasing-F1 promotion criterion, so no additional combination revision is run. Two fixed previews per removal control were inspected and reviewed beside the run; no-RGB shows patch-like corrections and border artifacts in those examples.

Development selection: retain the internal-feature CNN for the data-size/resolution stage, alongside R0. Its three-seed mean accuracy matches C1, uses fewer parameters and preserves slightly higher boundary F1. This is a development decision before new test inference.
