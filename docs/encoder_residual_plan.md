# Encoder feature residual counterparts, fixed 2026-11-25

The user approved a single-stage S0 comparison first, followed by F2, F3, SS1,
P2, P3 in that order. P1/P2 in the request was explicitly clarified to mean
the existing P2/P3 names. Prefix E identifies encoder counterparts, never old results.
Each method uses seeds 320112, 320113, 320114 before advancing to the next method.

Only 577 fitting and 218 scene-disjoint validation images are used. No new test,
all-795 refit, LoRA, sample-efficiency, ensemble, or architecture search is included.
Validation reuse makes this an exploratory comparison, not a fresh blind benchmark.

## Predeclared features and architecture

All features are the actual pixel_encoder.forward outputs at one-based ViT block
numbers, after the official optional normalization and removal of class/register
tokens, before UniDepth stage aggregation and decoder processing. The complete
frozen foundation still produces D0. No GT enters feature extraction or inference.
All selected encoder outputs have 384 channels and the same native 31x41 grid.
They are not claimed to be native multiscale features.

| Order | Candidate | Encoder blocks | Feature processing |
|---|---|---|---|
| 1 | ES0 | 12 | Same single 16-channel projection and residual CNN as S0 |
| 2 | EF2 | 4,12 | Flat projections 8+8 to common 240x320 grid |
| 3 | EF3 | 4,8,12 | Flat projections 6+5+5 to common grid |
| 4 | ESS1 | initial 2; pairs (3,4),(7,8),(11,12) | SS1 learned modules, widths 32/24/16, CBAM, common residual CNN |
| 5 | EP2 | 4,12 | P2 graph, projections 8+8; constructed grids 31x41 and 248x328 |
| 6 | EP3 | 4,8,12 | P3 graph, projections 6+5+5; constructed grids 31x41,124x164,248x328 |

For EP and ESS1, bilinear interpolation AFTER the learned channel projection
constructs grids matching the prior decoder experiment. It avoids expanding 384
channels on large grids. ESS1 keeps genuine same-native-scale distinct feature
pairs, attention from current and existing features, learned up-convs, double
convolutions, BatchNorm, LeakyReLU and CBAM. It is a further task adaptation of
StyleSketch (official reference 0214b46d51a7cd6e50674b4e4461adb3f720aedc), not an
official encoder adaptation or an original invention. The projection/init widths
and downstream CNN match existing counterparts, but projection input widths,
feature content/resolution and ESS1 feature sources differ. Attribute results to
complete candidates, not to encoder/decoder alone under perfectly matched capacity.

Inputs to the common CNN remain RGB and log(D0)/2 plus 16 projected feature channels;
output is additive delta D, restored to the original grid, D=max(D0+delta D,.001).
Use the existing exact RGB/depth/GT/mask disk cache read-only after SHA checks.
No new feature scaling, calibration or per-validation normalization is fitted.

## Training and validation

AbsRel loss, AdamW lr 3e-4, weight decay .001, batch 8, cosine schedule 80 epochs
to 1e-5, validation every 2 epochs, minimum 24 epochs and patience 5 validation
checks, gradient norm clip 1, FP16 AMP and no augmentation. Do not reuse F3's
68-epoch refit duration. Seeded batch order and numerical scoring are reused.
Validation runs in eval mode and must not change buffers. A discarded fitting-only
3-step engineering check verifies zero initial residual and gradients. Every
candidate's source/config/data identities are frozen before any formal training.

Course crop/range and boundary diagnostics remain unchanged. Report AbsRel, RMSE,
delta1, macro boundary F1, boundary-band AbsRel, parameters, time and peak VRAM.
Reuse old validation results only after split, protocol, source and checkpoint
audits. Compare each encoder variant with its decoder counterpart and include B0
and current F3 reference; distinguish seed SD from scene-bootstrap uncertainty.
Choose by 3-seed mean validation AbsRel; within 0.2% relative prefer fewer trainable
parameters, then lower AbsRel. Do not overwrite the old selection/freeze.

Preserve best/final weights, full curves, per-image scores, failures, config and
source hashes. Inspect best-to-final deterioration, not only selected minima.
Fixed qualitative examples: NYUv2 IDs 5,598,1440,226,246,80, seed320112.
Queue stops on a failed job; an explicit resume preserves RNG and optimizer state.
No silently skipped failures or architecture changes based on validation outcomes.
