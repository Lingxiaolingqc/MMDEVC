# Residual controlled study — 2026-11-18

Status: experiments completed on 2026-11-18; see [final results](../results/residual_final_test.md) and [record manifest](../results/residual_study_manifest.json). This file retains the authorized design and stage decisions. Qinchuan XIAO authorized Codex to implement, run and analyze the experiments, preserve every training record, and clean redundant generated artifacts. This plan supersedes an edge-only input comparison. It does not automatically extend to LoRA or ensembles.

## Research question and fixed protocol

Can a small supervised refiner improve frozen UniDepthV2 Small metric accuracy, especially around depth boundaries, and which inputs/architecture/output/loss actually contribute?

- Reuse `configs/edge_association_split.json`: 577 fitting images / 218 validation images, separated by scene; their union is the official 795 training images. The official 654 test images are excluded from design selection.
- The test set has been inspected in earlier project work. Future frozen evaluation is not a previously untouched blind test; disclose the history.
- Freeze the original checkpoint and inference settings. No LoRA weights are used in this study.
- Course evaluation: original 480x640, crop [45:471,41:601], valid GT [0.001,10] m, prediction clip [0.001,10] m, equal per-image averages, no per-image GT scale alignment. Validation AbsRel is the primary selection metric; also report RMSE and delta1.
- Boundary diagnosis: relative depth jump 10%, 2 px matching tolerance, 10 px distance cap, original resolution. Report precision/recall/F1, capped distance, and depth errors inside/outside a fixed 2 px GT-boundary band. GT boundaries may supervise training but never enter inference features.
- Source/manifest/checkpoint identities, package versions, timings, memory and all run outcomes must be recorded. Cached inference does not erase the deployment cost of a second flip pass or internal features.

## 1. Base predictions and R0

Let D0 be the original prediction, Df the horizontally flipped input prediction restored to original coordinates. A single global scale s is fitted only on fitting images, with equal image weighting of the AbsRel objective.

| ID | Base prediction |
|---|---|
| B0 | D0 |
| B1 | s * D0 |
| B2 | 0.75 * D0 + 0.25 * Df |
| B3 | s * (0.75 * D0 + 0.25 * Df) |

The 0.25 weight is a fixed historical reference, not a claimed optimum. Fit separate s for B1/B3. Evaluate the four bases, then attach the same refiner to each, using RGB + log(current base). Distinguish flip averaging from supplying Df as a feature.

R0: RGB + log(D0), 4 channels; three 3x3 Conv/GELU hidden layers of width 32, then a 1-channel output; last layer initialized to zero; additive meter residual, positive prediction clamp. Feature resolution 240x320; resize the predicted residual to 480x640 before supervision/evaluation. Loss: per-image masked AbsRel, averaged across images. Model selection uses the same course evaluation for every candidate.

Implementation details fixed before formal R0 training: depth feature logs are divided by 2 (as in the historical input scaling); RGB is mapped to [-1,1]. The training objective uses the same [0.001,10] prediction clip as the course metric. Scale fitting minimizes the full-pixel course objective on fitting only, with one scalar constrained to [0.25,4]; hitting a bound is an error requiring an explicit recorded decision. Edge calibration uses the fixed crop without GT masks, 5% RGB / 2% predicted-depth Canny density targets; magnitude normalization uses fitting RGB q95 / predicted-depth q98, clipped to [0,5]. Full-resolution edge maps are area-resized, so binary maps become occupancy fractions at 240x320.

## 2. Input comparisons (all other R0 settings fixed)

| ID | Change | Question |
|---|---|---|
| depth_only | Remove RGB | Is appearance useful? |
| flip | Add log(Df) | Is a second prediction informative? |
| confidence | Add transformed official confidence | Does model confidence help correction? |
| coordinates | Add normalized x,y | Do spatial priors help within NYUv2? |
| frozen_features | Add a fixed frozen depth feature tensor, learned 1x1 compression to 16 channels | Do internal representations help? |
| rgb_sobel | Add continuous RGB Sobel magnitude | Does explicit appearance gradient help? |
| rgb_canny | Add RGB Canny mask | Does a sparse contour help? |
| depth_sobel | Add Sobel magnitude of log(D0) | Does predicted geometry gradient help? |
| depth_canny | Add Canny on log(D0) | Does sparse predicted geometry help? |
| flip_disagreement | Add q=log(Df)-log(D0) and abs(q) | Is explicit disagreement easier to use than raw Df? Compare with flip as well as R0. |
| joint_edges | Add both RGB and depth Sobel | Is combining appearance/geometry helpful? Compare R0, each alone, and both. |

Edges/disagreement/coordinates are deterministic representations, not new sensors or GT. Confidence is not assumed to be calibrated correctness probability. Select detector thresholds using fitting-side density only, with the selection rule and actual thresholds recorded before validation scoring. Normalization statistics use fitting data only. Df always means restored original orientation.

Optional second-priority candidate: fixed multi-scale log-depth high-pass features, only if earlier errors motivate it; record the decision before its validation run. No unbounded feature search.

## 3. Separate architecture, output and loss groups

Each group initially uses the common R0, not the previous group's winner.

- Architecture: pixelwise 1x1 MLP; R0 CNN; small U-Net. Match trainable parameter counts approximately when practical and publish exact counts, FLOP/latency limits, and resolution. U-Net is a CNN. Differences under this common budget do not prove architecture-wide superiority.
- Output: additive meters; unrestricted log multiplicative residual; bounded log multiplicative residual (gamma=0.5 historical reference). Zero output initialization always preserves the base. A fixed scalar in front of an otherwise unrestricted additive head is not treated as a distinct capacity claim.
- Loss: AbsRel; AbsRel + 0.1 * valid-neighbor log-depth gradient L1; GT-boundary-weighted AbsRel (weight=2 in the fixed boundary band, weight=1 elsewhere, normalized per image). These fixed starting values are not claimed optimal. Do not match every RGB texture edge to a depth discontinuity.
- Historical bundle: RGB, log(D0), log(Df), log(calibrated flip base), transformed confidence, x,y = 9 channels; U-Net widths 16/32/64/96; bounded multiplicative output. Re-run under the new split/protocol as an overall reference. It changes several factors and cannot establish individual component contributions.

## 4. Training budget and selection

- Screening seed 320112; confirmation seeds 320112/320113/320114. Shared deterministic image order per seed, same fitting/validation IDs and validation schedule.
- AdamW lr=0.0003, weight_decay=0.001, batch_size=8, 80 epoch cap, cosine schedule to 0.00001, validation every 2 epochs, earliest stop after 24 epochs, patience 5 validation checks, gradient-norm cap 1. No random augmentation in the initial one-factor screening.
- AMP is allowed and recorded; nonfinite predictions/loss cause a failed run, never silent pixel removal. A limited smoke run validates engineering only and is labeled separately.
- Shortlist at most three candidates after screening using validation AbsRel and failure/efficiency evidence. Re-run shortlisted candidates and R0 across three seeds. Preserve negative or statistically ambiguous outcomes.
- Combine only motivated components, then remove one component at a time. Limit the first combined design to one documented configuration, with at most one predeclared revision justified on validation. Do not present marginal single-seed changes as reliable wins.
- At most three confirmed methods (plus frozen model) receive nested fitting-size 10/25/50/100% comparisons. Use a deterministic scene-group ordering, publish actual counts, keep validation fixed, refit scales on each fitting subset, and report epochs, updates, image exposures and wall time. Reuse compatible full-data runs.
- Confirm 240x320 versus 480x640 input for R0 and the selected method if needed to support the boundary claim. Report the compute increase.
- Freeze methods, seeds and training duration using development evidence; refit on all 795 without test early stopping, evaluate all 654 test images for the predeclared methods. Report seed-level results, scene-aware paired uncertainty and per-image tables.
- No guaranteed improvement: if it fails to improve, report that, explain the bounded attempts, and preserve the evidence.

## 5. Records and storage policy

Code: `src/p3_depth/residual/`; runnable entry points: `scripts/part3_improvement/`. Configuration: `configs/residual_study.json`. Generated data and runs: `outputs/part3_improvement/residual_study_20261118/`. Small tracked summaries: `results/residual_study_status.md` and later stage reviews.

For every run, including failed/stopped pilots, retain config, data/source hashes and split IDs, seed, complete training/validation history, selected checkpoint step and hash, per-image validation metrics, aggregate metrics, timing/memory, status/error, and a short interpretation. Preserve best checkpoints (small refiners), required final weights, representative comparisons and source snapshots. Store measured outputs separately from planned claims. Running jobs keep one atomic resume checkpoint.

Clean only inside the new study's resolved absolute root unless separately justified. Before deleting, record path, bytes, hash, reason and reconstruction recipe in `cleanup.jsonl`; mark success after deletion. Prefer not creating duplicate material. Delete superseded resume checkpoints only after successful completion and verification of the best checkpoint/records. Delete large duplicate predictions/caches only when no pending experiment needs them and regeneration inputs/code are retained. Never delete raw datasets, official weights, user originals, existing unrelated outputs, or sole evidence of a result. Never delete a running job's state.

## 6. Deliverables and continuation

1. Reproducible scripts/configs and recorded training runs.
2. B0-B3/R0, single-factor, seed, combination, learning-curve and final-test tables with matched protocols.
3. Quantitative and visually verified qualitative analysis, failure cases and costs; distinguish historical evidence, new measurements and hypotheses.
4. A concise report-ready Residual section and current status/next command. Keep LoRA independent until this study is interpretable.

Execution authority was given on 2026-11-18. The agent runs the experiments and analysis; no further per-run confirmation is needed within this scope. A useful stage is not the same as completion of the entire goal.

## 7. Recorded post-screening decisions

All 22 seed-320112 screening runs completed on 2026-11-18. Before running the other seeds, `confirmation_selection.json` fixed R0 plus three candidates: frozen internal features, small U-Net, and the historical bundle. The internal-feature candidate had the strongest primary accuracy result. U-Net isolates architecture. The bundle provides an accuracy/boundary/cost tradeoff reference, not causal evidence for its individual components. Single-seed edge effects and negative findings remain in the full matrix.

The data-size study is separately fixed in `data_size_design.json`: 58/145/289/577 images in nested whole-scene subsets (18/47/95/187 scenes), unchanged 218-image validation, seed 320112, 2000 updates of batch size 8, exactly 16000 image exposures, evaluation every 200 updates, no early stopping, cosine schedule over 2000 steps, score the final iterate. Compare R0 and one development-selected Residual design; include frozen and subset-calibrated scalar references. This is an exploratory fixed-compute comparison, not the optimum attainable at each data size. It must not reuse differently budgeted screening scores as if they were matched.

`run_residual_study.py --action confirm --resume` executes the recorded seed shortlist with the original numerical training protocol. `run_residual_jobs.py --jobs <immutable JSON files> --resume` executes subsequently recorded combinations, ablations, data-size, resolution and refit jobs. Job definitions and IDs are immutable. Final refits require the hash of `frozen_selection.json`, all 795 development IDs and an empty validation-selection set. All-development training duration is declared in advance; the cosine horizon can remain 80 epochs when a shorter selected duration is used.

The initial screening keeps the validation-selected deployment weights and complete records. Subsequent training also keeps a separate last-iterate checkpoint; fixed-final data-size/refit results always score that checkpoint. Completed resume-state deletion verifies the retained weights and is logged. Expanded package/helper provenance and an OS-released study lock were added after screening without changing model/loss/update mathematics.

## 8. Combination decision and remaining controlled follow-ups

The three-seed C1 combination (internal features plus small U-Net) does not establish an improvement over the internal-feature CNN: mean validation AbsRel 0.099889 versus 0.099848, with slightly lower boundary F1. Its zero-feature and no-RGB controls are retained. No-RGB fails the predeclared non-decreasing-F1 promotion condition, so the optional combination revision is not expanded. `configs/residual_followup_selection.json` records the internal-feature CNN as the main development-selected method, with R0 as reference, before data-size, resolution and new test inference.

The fixed-compute size budget means 2000 attempted updates and exactly 16000 image exposures. AMP overflow can skip a weight update; all such events are recorded and counted separately. This clarifies the existing schedule without changing it. Do not describe it as exactly 2000 successful weight changes. Subset-only scalar controls use the same labeled images but are not neural matched-compute runs.

The resolution sensitivity uses seed320112, 240x320 versus 480x640, with unchanged model and epoch protocol for R0 and the internal-feature CNN. The default confirmed method is 240. Before these runs, promotion of 480 was limited to at least 1% relative lower AbsRel and +0.002 absolute boundary F1 on the selected method, followed by the other two declared seeds. Otherwise retain 480 as a one-seed sensitivity result. No further architecture/loss search is triggered.

## 9. Final selection before new test inference

Resolution480 failed both recorded promotion thresholds; keep240. Final methods are R0 and the internal-featureCNN, each with seeds320112/320113/320114. Refit all795 images for the median development-selected duration: R0=2epochs (2,6,2), internalCNN=76epochs (76,74,80), keeping the80-epoch cosine horizon. No final checkpoint selection from validation/test. Compare against all four B0–B3 references, fitting scalar calibration on all795 only. Freeze exact methods, durations, sources and45 completed development results before any new test cache. Representative displayed checkpoint is predeclared seed320112; report each seed and mean scores, not prediction ensembling. Preserve equally disclosed ranked best/worst and fixed first/middle/last test cases after evaluation.
