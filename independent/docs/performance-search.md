# Performance objective updated on 2026-11-11

The user clarified that the objective is the strongest measurable performance within the course rules. Explore competing directions and select using development results. The existing Direction A implementation is an initial experiment. Its 18.23% temporal-proxy reduction leaves the mandatory NYU metric-depth accuracy unchanged. Its normal-speed visual effect is subtle.

This instruction supersedes the earlier design's choice to retain Direction A and its blanket exclusion of model adaptation. Preserve the completed reference runs and compare each new candidate with its own unchanged model. Continue using independent feature branches.

The user then supplied the Part 2 screenshot and clarified the teacher's emphasis on gains over a SOTA reference within the same category. The course names UniDepthV2 ViT-S or Depth Pro for metric depth, and VDA-Small for video. Use our reproduced UniDepthV2 ViT-S as the primary quantitative reference. Improving its NYU scores directly addresses the metric-model/improvement comparison in Part 2. Describe broader published methods as additional comparisons until matched results have actually been reproduced.

For each comparison, record the task, input modality, checkpoint/model size, data split, output size, crop, masks, metrics, and adaptation/training data. A calibration fitted with NYU development ground truth is a domain-adapted setting. Label that setting explicitly when comparing with zero-shot pretrained models. The video has no supplied depth GT, so its temporal diagnostics cannot establish metric-depth SOTA.

## Baseline and ground truth

- Main quantitative reference is unchanged RGB-only UniDepthV2 ViT-S on the course NYU split. Existing test AbsRel 0.11454576, RMSE 0.39888770 m, delta1 0.89221783.
- NYU ground truth is the supplied filled metric depth, derived from Kinect measurements. All candidates share the required output resolution, crop, masks, and equal per-image aggregation.
- Video reference is unchanged DA V2-Small, with VDA-Small as a separate trained video comparator. The supplied video has no ground-truth depth. A change in its predictions cannot by itself establish accuracy improvement.
- Stereo uses the 15 Middlebury quarter-resolution pairs and supplied disparity GT as specified by the brief.
- Main inputs are the teacher-specified public datasets. Tokyo is an optional example from the same video repository and was used solely for temporal development. NYU MAT transport used a mirror, with the recorded file hash and official split. No private teacher-specific data copy has been verified against our file.

## Competing experiments to prioritize

1. Establish NYU development predictions and error breakdowns for global scale bias, boundaries, and large residual regions. Use only the 795 development images for fitting and method selection. The 654 test images have already been evaluated for the baseline, so later results should disclose that history.
2. Compare three inexpensive families against unchanged UniDepthV2 on a held-out development subset. Global scale calibration follows the brief's Direction B. Flip/multi-resolution test-time averaging tests prediction variance. RGB-guided refinement tests boundary error and texture transfer. Measure accuracy and inference cost for each family before combining modules.
3. If these gains are small, investigate lightweight learned residual refinement or limited adaptation using development data, with a fitting/validation split. This changes the adaptation setting and must be disclosed beside the original pretrained reference.
4. Review recent strong Small-model methods and available official code/weights. Select a named comparison and reproduce the same course protocol before making a state-of-the-art claim. DepthART (2026) is a literature candidate. Its paper's zero-shot NYU delta1 and throughput use a different reported setup and are not directly comparable to our run.
5. Choose the final improvement from the measured development accuracy/runtime tradeoff. The brief requires one final improvement, while allowing alternative technically sound methods. Preserve unsuccessful experiments as ablations where they explain the choice.

## Demo correction

The script scripts/diagnose_fusion_difference.py creates a separate six-FPS, quarter-speed diagnostic. It preserves all 70 source frames and displays original RGB, DA V2, fusion, and a fixed-scale absolute prediction-change map. The change map is magnified with saturation at 2% of the original display range. The preview uses the frame with largest mean absolute change. It supports inspection of where the method acts. True depth accuracy on this video remains unknown.

The next model-comparison experiments are proposed here and have not yet run. This update does not claim a new NYU accuracy gain or a reproduced SOTA result.
