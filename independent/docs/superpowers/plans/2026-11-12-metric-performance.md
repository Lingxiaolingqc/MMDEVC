# NYU metric performance implementation plan

The user authorized execution of docs/performance-search.md on 2026-11-12. Use executing-plans inline, with one final fresh review. Reuse the existing independent checkout and feature branch.

Goal: select and evaluate the strongest measured lightweight improvement over unchanged UniDepthV2 ViT-S using the teacher's NYU protocol.

Architecture: cache unchanged and horizontally flipped predictions on the 795 development images. Divide these images by whole scene into fitting and validation sets using a declared deterministic seed. Compare global calibration, equal-resolution flip averaging, edge-aware filtering, and a lightweight learned residual refiner if initial gains are limited. Choose using development validation AbsRel, then freeze and run all 654 test images once for the selected final method and its relevant ablations.

Constraints: existing baseline checkpoint and input size remain fixed. RGB-only inference. Same output size, crop, finite GT mask, clipping, and equal-image metrics. Fit using development labels only. Describe supervised refinement/calibration as NYU adaptation. Preserve earlier test history. No claims of universal SOTA based only on one reference comparison. All source/results live under independent/ on the feature branch.

Review focus: scene overlap and test-ID leakage; resizing/flip reversal; invalid or empty depth support; cache identity and resumability; clipping and fitting objective correspondence; runtime including full network inference.

- [ ] Cache all development predictions and record scene-disjoint fitting/validation IDs, source/checkpoint identities, preprocessing, and exact inference settings. Verify no scene overlap or test IDs.
- [ ] Add tested calibration/refinement primitives. Verify a known scale error, invalid inputs, flip reversal, identity transform, finite outputs, and preservation of the course protocol.
- [ ] Evaluate unchanged, flip averaging, calibration, and RGB-guided filtering on the same validation images. Record unsuccessful candidates and measured inference costs.
- [ ] If evidence warrants, fit a small residual CNN on development fitting scenes, select its checkpoint only on reserved validation scenes, and compare its contribution with the simple candidates. Record weights, training curves, seed, and parameter count.
- [ ] Freeze the selected method/config before final test. Evaluate baseline and selected ablations on all 654 test IDs using the unchanged protocol. Generate paired per-image changes and deterministic uncertainty estimates.
- [ ] Inspect same-image RGB/GT/baseline/improved/error examples, report genuine failures, and create a readable quantitative comparison. Update the report and reproduction commands around the final improvement.
- [ ] Run appropriate tests, review once, fix material findings, commit and push only the independent feature branch. Report actual gains, adaptation setting, and limitations.

Ruling: multi-resolution inference is excluded from the primary search to respect the instruction to keep resolution fixed between a method and its improvement. Horizontal flip averaging preserves the model/input resolution. The previous temporal experiment remains historical evidence.
