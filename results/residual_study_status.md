# Residual study status

Completed: 2026-11-18. All 51 formal training runs, six frozen all-development refits included, and all ten 654-image test evaluations are complete. One engineering smoke run is recorded separately.

## Result

The selected internal-feature CNN refines frozen UniDepthV2 Small predictions. On the full NYUv2 test split, its three-seed mean AbsRel is **0.090448**, versus **0.110776** for the original model (18.35% relative reduction), **0.102179** for global-scale calibration and **0.101263** for the basic RGB/depth CNN. RMSE decreases from 0.392817 to 0.362062 metres.

Boundary-band depth error also improves. Additional boundary F1 beyond the basic CNN is not established: its paired scene-bootstrap interval crosses zero. Visual inspection does not establish recovered fine structures. There are 433 improved and 221 worsened images versus the original model; failures remain in the report.

Start with the [report section](residual_report_section.md), [final test analysis](residual_final_test.md), or [Chinese reading/reproduction guide](../docs/residual_results_guide.md).

## Experiments and evidence

| Stage | Completed work |
|---|---|
| Foundation and baselines | Cached all 795 development and 654 test images; evaluated original, global-scale, flip and scale-plus-flip references. |
| Single-factor screening | 22 configurations for input, architecture, output and loss; [analysis](residual_study_screening.md). |
| Seed confirmation | Eight additional runs, completing three seeds for R0 and three candidates; [analysis](residual_seed_confirmation.md). |
| Combination and removal | Five additional runs; [analysis](residual_combinations.md). Simpler internal-feature CNN retained. |
| Label size and learning curves | Eight matched 2000-update runs on nested 58/145/289/577-image subsets; [analysis](residual_data_size.md). |
| Resolution | Two additional 480-resolution controls failed the predeclared promotion rule; [analysis](residual_resolution.md). |
| Frozen final refits | R0 and internal-feature CNN, three seeds each, all 795 development images. Fixed durations: 2 and 76 epochs. |
| Final test | Four baselines plus six independent neural runs, all 654 test images; [analysis](residual_final_test.md). |
| Deployment | Saved-weight input/batch parity and isolated latency measurement; [runtime](residual_runtime.md). |

Design selection used 577 fitting / 218 scene-disjoint validation images. Methods, seeds, durations and numerical sources were frozen before the new test inference. Earlier project test exposure is disclosed; this is not a never-seen blind test. LoRA and historical DA video fusion are separate from this study. The original [research plan](../docs/residual_research_plan.md) is retained.

## Validation and retained records

All 27 focused correctness tests passed. The record audit verified all 51 formal trainings plus the smoke run: immutable source snapshots, retained checkpoint hashes, history lengths and evaluation row counts. Final evaluation audit verified all ten artifact sets, full sample lists, aggregates and 30 fixed raw-prediction rescoring checks. Fresh B0 exactly reproduces the prior saved course and boundary metrics.

Full evidence is local under `outputs/part3_improvement/residual_study_20261118/`. Each training retains configuration, source/data identities, complete history, selected weights, metrics and previews; follow-up jobs also retain final iterates. Final test predictions remain available. See the compact [delivery manifest](residual_study_manifest.json) for hashes and the run inventory. Code/configuration/compact reports are synchronized through Git; generated data and model weights require local copies or reproduction.

Removed 21 rebuildable input-cache files, totaling **12.168 GiB**, after all dependent experiments and audits completed. Metadata copies and deletion hashes are retained in `cleanup_manifests/` and `cleanup.jsonl`. Empty OneDrive folders remain where directory removal was denied. Original data, official weights, required trained weights, final predictions and earlier project outputs remain.

The final three internal-feature refits each recorded two AMP nonfinite-gradient events; these are retained in their histories. All test predictions were finite and no test image was omitted. A pre-training source-snapshot failure and its recovery remain in the run ledger.
