# SS1: one StyleSketch-inspired depth Residual candidate

Predeclared on 2026-11-20 before training. No width, layer, or component search.
Reference: kwanyun/StyleSketch commit `0214b46d51a7cd6e50674b4e4461adb3f720aedc`,
`sketch/stylesketch_utils/stylesketch.py` and `conv.py`. This borrows their modules
and fusion topology; it is neither an original fusion mechanism nor a full reproduction.

## Official operations retained

Same-resolution feature pairs projected separately by 1x1 convolution, then concatenated;
coarse-to-fine fusion; nearest-neighbor 2x resize followed by learned 3x3 convolution,
BatchNorm and LeakyReLU(0.2); Attention_block using both accumulated feature g and
current feature x, with separate Conv1x1+BN, sum, LeakyReLU, Conv1x1+BN+Sigmoid;
intermediate concatenation followed by two Conv3x3+BN+LeakyReLU blocks; final
concatenation followed by CBAM (shared avg/max channel MLP then 7x7 spatial attention,
including its BN with momentum 0.01). Internal attention Sigmoids are retained.
Official up_conv is resize+convolution, not transposed convolution.

## Task adaptations fixed before formal training

Frozen UniDepthV2 Small, official resolution_level=0, predicted camera, no GT alignment.
Native starting feature init: 256x31x41. Its existing trainable 1x1 projection produces32 channels.
Three native pairs (NCHW, independently computed actual module outputs):

| Level | Pair | Native shape per tensor | Projected pair / fused width |
|---|---|---|---|
|1|depth_layer.ups[0], process_features[1]|256x62x82|16+16 /32|
|2|depth_layer.ups[1], process_features[2]|128x124x164|12+12 /24|
|3|depth_layer.ups[2], depth_mlp[2]|64x248x328|8+8 /16|

depth_mlp[2] is the real LayerNorm+Linear depth feature, converted NHWC->NCHW.
It is related to ups[2] but is not a copied tensor. The two process_features outputs
are separate learned conditioned-feature branches. No fabricated duplicate levels.
Feature extraction must check all expected shapes, pair inequality, unchanged frozen
model parameters/buffers, and bitwise FP16 identity of prior init/depth predictions.

Widths32/24/16 and three levels replace StyleGAN's eight upsampling levels and much
larger channels. As in SketchGenerator, only the first two levels use conv_block;
the final attention/upsample concatenation goes directly to CBAM32. A task-specific
1x1 projection reduces it to16, then bilinear resize to240x320. These16 channels are
concatenated with normalized RGB3 and log(D0)/2 to feed the identical shared32-channel,
three-hidden-convolution GELU residual CNN. The zero-initialized output predicts
unbounded additive deltaD, bilinearly restored to480x640; D=max(D0+deltaD,0.001).
RGB enters the common CNN, not CBAM (unlike official final RGB concatenation).
No sketch output Sigmoid, sketch losses, discriminator, StyleGAN, or sketch weights.
More feature inputs and capacity than F3: compare complete candidates, not fusion alone.

## Fixed training and selection

577 fitting /218 scene-disjoint validation, seeds320112/320113/320114. No test data.
Same per-image course-clipped AbsRel loss, AdamW lr3e-4, weight decay0.001, batch8,
gradient clip1, FP16 AMP, no augmentation. Cosine schedule80 epochs to1e-5, evaluation
every2 epochs, minimum24 epochs, stop after5 evaluations without strict AbsRel improvement.
Use each seed's best validation checkpoint; retain final checkpoint and full curves.
Do not use F3's68-epoch future refit duration. BN statistics update on fitting batches
only, including the final partial batch; evaluation always eval()+inference_mode().

Primary selection: three-seed mean validation AbsRel between preselected F3 and SS1.
Within0.2% relative of the lowest score, prefer fewer parameters, then lower AbsRel.
P3/G3/B0 are references. Report sample SD, scene-cluster paired95% bootstrap intervals
(5000 draws, seed320118), per-seed deltas and all218 paired images. These are exploratory
reused-validation results, not multiplicity-corrected causal evidence or unseen-test claims.
Boundary F1 and boundary-band AbsRel are diagnostics, with no post-hoc F1 veto.
Existing depth crop/range, max log-depth neighbor edge jump>ln1.1,2px tolerance unchanged.

Visuals fixed before training: sample IDs0005,0598,1440,0226,0246,0080 (one-based).
Also show largest gain/regression selected by a disclosed post-hoc quantitative rule.
Use identical color/error scales and the same seed320112, not averaged predictions.
Compare costs with exact timing boundaries; cached-refiner time excludes UniDepth
feature construction. Record extraction separately and benchmark F3/SS1 in one session.

Before formal training: official block output/gradient parity, real-feature shapes,
all-module gradient reachability after the zero-head warm-up, initial deltaD=0,
fixed foundation and validation BN-buffer immutability. Engineering weights are not
used to initialize formal seeds. Design/source hashes freeze before first formal update.

Retain configuration, source archive, feature source/hash, best/final weights, curves,
per-image best/final scores, logs and failures. Separate output stylesketch_20261120;
never overwrite prior selection. Stop after Residual selection: no sample-efficiency,
LoRA search, all795 fitting, or new654 test evaluation.
