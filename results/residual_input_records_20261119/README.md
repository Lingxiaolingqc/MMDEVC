# Residual input combination experiment records — 2026-11-19

This archive contains six single-seed screening runs and four additional confirmation runs (10 formal runs), plus two engineering smoke runs. No new method passed all predefined confirmation conditions, so no new full-development refit or 654-image test was performed. The previous frozen S0 result remains unchanged.

Read the [stage-by-stage results](../residual_inputs_summary_20261119.md) and [experiment protocol and commands](../../docs/residual_inputs_plan_20261119.md) first. All new accuracy numbers are from the same 218-image, 62-scene validation set; fitting used 577 images. Three-seed summaries average separately trained models' metrics, not ensemble predictions.

## Contents

| Location | Evidence |
|---|---|
| `jobs/`, `followups/` | Immutable configurations, training histories, results, per-image scores, checkpoint identities and run provenance |
| `analysis/screen/` | Six-candidate screening comparisons and plots |
| `analysis/development/` | Three-seed confirmation tables, paired scene intervals and training curves |
| `analysis/` | Matched visual comparisons and `visual_review.json`, which identifies the figures actually inspected |
| `benchmarks/` | Complete RGB-to-depth timing and live/cache parity checks |
| `reference_validation/` | Recomputed original S0 reference metrics |
| `screen_selection.json`, `conditional_decision.json`, `confirmation_decision.json` | Recorded selection, conditional stopping and confirmation decisions |
| `engineering_verification.json`, `engineering_events.jsonl`, `delivery_audit.json` | Implementation checks, failures and delivery audit |
| `cleanup.jsonl`, `retained_cache_metadata/`, `cache_metadata_locations.json` | Rebuildable-cache removal records and preserved metadata |
| `index.json`, `source_snapshots.zip` | Export hashes and exact source snapshots associated with runs |

M2 improved single-seed overall accuracy but failed this study's boundary-F1 guard. Three-stage fusion, edge-plus-multistage fusion and multibranch zero-input controls therefore were not triggered. These are unperformed experiments, not negative results. I1 and I4 were not selected for additional seeds under the fixed two-candidate limit; they have not been disproven.

## Integrity and storage

`index.json` describes 246 exported record/media files (36,095,999 bytes, about 34.42 MiB). It includes both the source and exported SHA-256. Training logs are exported byte-for-byte with `.txt` extensions. In exported `provenance.json`, only `git_status` is omitted; all other fields are preserved. Where export bytes differ, the index records both hashes.

The 167,551-byte `source_snapshots.zip` stores a manifest and deduplicated source blobs identified by SHA-256. It preserves the versions used by individual runs; these snapshots may differ from later command-line or documentation housekeeping. The ZIP's hash is recorded in the index. The local `.gitattributes` disables text newline conversion so indexed bytes survive Git checkout. This README, `.gitattributes` and the index itself are not counted among the 246 indexed files.

Dataset arrays, official model weights and training weights are not committed. All 24 best/final checkpoint files from the 12 runs remain locally under `outputs/part3_improvement/residual_inputs_20261119/followups/`; their hashes were checked before and after cleanup. The earlier frozen study and its caches were preserved.

The completed cleanup removed 18 rebuildable cache artifacts totaling 18,095,018,874 bytes (16.85 GiB). Their metadata and deletion hashes remain recorded. To rerun cache-dependent work, regenerate these caches using the preparation command in the protocol. Redundant resume-state cleanup is logged separately.

The final-refit/test command paths are conditional experiment code. They were not exercised as final experiments in this study and must not be reported as completed test results.
