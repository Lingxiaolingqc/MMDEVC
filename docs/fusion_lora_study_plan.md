# Independent Residual fusion and LoRA selection

Current scope, clarified by Qinchuan XIAO on 2026-11-19: complete the three feature-fusion comparisons, freeze Residual, independently select and freeze LoRA, then stop. The comparison between the two selected methods, matched-data-size runs, all795 refits and new654 test evaluations are deferred. The earlier broader plan is retained below only as a future design. This is a new study; previous decisions and results remain immutable. No prediction ensemble is included.

## Fixed protocol and evidence limits

Reuse `configs/edge_association_split.json`: 577 fitting / 218 scene-disjoint validation; final official test 654. Retain the existing course crop, [0.001,10] m clipping, predicted camera intrinsics, image-macro metrics, boundary detector and no GT-derived inference features. Earlier test exposure and repeated validation use must be disclosed. Primary selection is validation AbsRel; RMSE, delta1, boundary F1, boundary-band AbsRel, parameter count, time and memory are secondary reports. The previous study's F1 rejection rule is not retroactively changed; it does not govern this new accuracy-first study. Best means best among the declared candidates, not a global optimum.

## A. Residual factorial comparisons

All six core cells receive seeds 320112/320113/320114. Fixed RGB/log-depth inputs, frozen UniDepthV2, additive residual, AbsRel loss, batch8, AdamW3e-4 and the previous 80-epoch schedule/early-stop rule. Feature projections have total width16: two stages8+8, three stages6+5+5. Stages mean initial decoder latent, optional middle decoder output, late decoder output; not arbitrary ViT layer indices.

| Family | Two stages: initial+late | Three stages: initial+middle+late |
|---|---|---|
| F: project, resize, concatenate once, CNN | F2 | F3 |
| P: progressively fuse at native stage resolutions | P2 | P3 |
| G: P plus spatial gates from the preceding fused representation | G2 | G3 |

Compare F/P for fusion design, P/G for gates, and 2/3 within every family for stage count. Report interactions, not just marginal averages. Previous M2 seed320112 may supply F2 only after exact model/input/config/output parity; other cells are new runs. S0 is a reused reference, not silently relabeled a factorial cell.

Because P has extra convolution parameters, add C2/C3: exactly the P2/P3 graph and parameters but resize all feature stages to the final grid before fusion. These two controls use seed320112 first, separating progressive spatial processing from parameter count. They are controls, not extra unrestricted search candidates. If a three-stage design is selected, train its same-graph zero-middle control at all three seeds; zero input retains parameter count but not effective functional capacity, and this limit must be stated.

Select the lowest three-seed mean validation AbsRel among the six cells and S0; within 0.2% relative AbsRel choose fewer parameters, then lower AbsRel if parameter counts are equal. Report paired scene-bootstrap intervals and seed SD separately. An interval crossing zero prevents a claim of established superiority, but does not trigger test-based reselection. Freeze the selected architecture and median best development epoch before independent LoRA selection. No new edge/loss/ensemble search.

## B. Independent bounded LoRA study

Use the same original UniDepthV2 Small checkpoint, raw RGB, resolution_level0, predicted camera, objective and fitting/validation split. Train only low-rank adapters; original weights stay frozen. Preserve official inference preprocessing and postprocessing, verify zero-adapter parity, and verify nonzero adapter gradients. No frozen inference context may silently suppress gradients.

Declared search: attention linear layers in either the pixel decoder or the final four ViT blocks; ranks4/8; learning rates1e-4/3e-4; alpha=rank, dropout0. Eight cells, seed320112, 2000 attempted updates, effective batch8, evaluation every200 updates. Microbatch/accumulation chosen by engineering preflight and fixed before formal runs. Preserve all zero-gradient, overflow and failed runs. The two lowest validation-AbsRel candidates receive the other two seeds under the same budget; choose lowest three-seed mean (same 0.2% simplicity tie rule). Freeze target paths, rank, learning rate and median best update count. No rank/location extension after seeing test.

## C. Matched labeled-data comparison — DEFERRED, do not execute in this stage

Reuse the exact nested whole-scene fitting subsets from the previous `data_size_design.json`:58/145/289/577 images (18/47/95/187 scenes). Do not redraw subsets or retune either design for each size. Compare the frozen Residual and LoRA designs at all four sizes and all three seeds. Same IDs/order/seed, 2000 attempted updates, effective batch8,16000 image exposures, no augmentation, fixed2000-step cosine horizon and final-iterate primary score; retain best-validation checkpoints only as diagnostic records. Learning curves every200 steps show overfitting or incomplete convergence. Reuse a prior run only when every numerical/data/budget setting agrees.

This controls updates and exposures, not wall time or FLOPs; report those costs. Validation results are developmental. Freeze all data-size checkpoints before their common654-image test evaluation; do not use test to choose sizes, checkpoints or architecture. Also refit each selected design on all795 with its frozen development-selected duration, three seeds, then score all654. The latter is same-label optimized-method comparison, not matched training compute.

## Delivery and storage

Keep per-run immutable configurations, code/data/weight identities, complete histories, best/final weights or adapter-only weights, per-image tables, paired scene intervals, fixed and ranked qualitative comparisons, and measured costs. Report all six Residual cells, controls, LoRA cells and repeated seeds directly in Chinese, with repository records. Retain failures and ambiguous improvements.

For the current narrowed scope, deliver only the Residual ablation/selection results and the independent LoRA search/selection results. Report each selected validation configuration and its uncertainty; no head-to-head winner between Residual and LoRA, data-size table, full-development refit or new test claim is produced now. Unexecuted commands for those later stages are not evidence of completed experiments.

Code belongs under `src/p3_depth/adaptation_study/` and experiment entry points in `scripts/part3_improvement/`. Outputs are isolated under `outputs/part3_improvement/fusion_lora_20261119/`. Only remove hash-recorded rebuildable caches or redundant completed resume state after all dependencies finish. Preserve prior frozen studies, original data, official weights and unrelated README/stereo edits. Commit/push is already authorized. No external messages or PR merge is requested.

## Engineering choices fixed before formal training

Both LoRA locations passed exact zero-adapter parity and finite, nonzero gradients over five engineering updates. Effective batch8 can run as microbatch8 (peak allocated approximately3.64/4.02GiB). AMP initial scale128 is fixed: the default65536 overflowed camera gradients; scale1 was finite but early tiny adapter-A gradients underflowed. All attempts remain recorded. This is numerical preflight on fitting images, not validation-based architecture selection. This original five-update preflight was insufficient; the numerical revision below supersedes its training-gradient claim. Official inference remains unchanged; camera attention remains included in decoder targets.

## Selection-only execution

Run `scripts/part3_improvement/run_adaptation_selection.py` using the repository CUDA Python environment. It executes the fixed Residual matrix and controls, freezes Residual, runs the LoRA engineering smoke and bounded search/confirmation, freezes LoRA, then generates independent analysis figures and matched validation cases. It stops on a failed stage and preserves its records. `--wait-for-current-study` waits for an existing study process before continuing; it does not launch concurrent GPU training.

The sequence contains no matched-data-size run, all-development refit, new test evaluation or Residual-versus-LoRA comparison. Individual runner options for these deferred stages are implementation scaffolding, not authorization or completed evidence. `selection_progress.json` distinguishes completed experiments from the subsequent visual review and archival report.

## LoRA numerical revision before restarting the formal grid

The first original-protocol decoder run was aborted and excluded: normalized camera-ray z rounded to 1, whose acos backward derivative is infinite. AMP scale collapsed and validation stalled. The incomplete run, checkpoint hashes, source snapshot, losses and failure diagnosis are preserved; it is not a valid candidate result.

All restarted grid and confirmation jobs use the same finite_rays_v2 protocol and distinct run IDs. A model-instance training-only ray embedding uses the exact official forward operations and acos values, but bounds the acos derivative denominator by sqrt(machine epsilon) at numerical poles. This is a surrogate endpoint gradient, not the exact mathematical derivative there. No official files, inference values, target layers, data split, objective or search budget change. No-grad inference calls the original ray embedding. AMP starts at128 with normal adaptive backoff; any collapse below1 aborts instead of silently producing a stalled search. Record skipped and attempted updates separately.

Before this restart, run preflight_lora_study.py --tag finite_rays_v2_diverse --steps 200: both locations, ranks4/8, the declared higher learning rate3e-4, and 200 varied fitting batches per cell. Verify zero-adapter parity, finite gradients after normal backoff, every adapter receiving a nonzero gradient and unchanged base weights. Engineering weights are discarded. This numerical correction is common to every LoRA candidate, not a validation-selected improvement.
