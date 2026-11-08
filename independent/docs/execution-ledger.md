# Execution ledger - plan: docs/superpowers/plans/2026-11-08-direction-a.md

User approved implementation on 2026-11-08. Base: `965a5e6`. Spec: `docs/superpowers/specs/2026-11-08-direction-a-design.md`.

## Pre-flight

| Producer -> consumer | Contract | Finding |
| --- | --- | --- |
| Task 1 -> 2, 4, 5 | Ordered HWC uint8 RGB Frame with index/timestamp | Consistent; source timestamps retained where decoder exposes them. |
| Task 2 -> 5, 6, 7 | HW float32 prediction, explicit relative/metric convention, cache metadata | Consistent; full VDA clip inference must preserve its temporal context. |
| Task 3 -> 6, 7 | Raw NYU metrics, stereo MAE/coverage | Independent from temporal fusion; no cross-image history. |
| Task 4 -> 5, 6 | Full-resolution forward/backward HW2 displacements and valid mask | Consistent; gather uses backward flow. |
| Task 5 -> 6, 7 | Causal state, diagnostics, float output | Consistent; one state per clip/config. |
| Task 6 -> 7 | Frozen configuration and run manifests | Consistent; final evaluation is separate from development selection. |

Ruling: Use the existing Python 3.12.6 interpreter in a project-local virtual environment rather than downloading 3.11 initially. Official UniDepth declares Python >=3.10; validate actual upstream inference before accepting this environment. Cost if wrong: a separate compatible environment may be needed.

Ruling: Maintain this tracked ledger and task briefs with native Python/PowerShell instead of the bundled bash bookkeeping helpers. The helper combines Windows `C:/...` Git paths with MSYS paths and loops on failed mkdir; it was stopped. All product tests and per-task verification remain required. Cost if wrong: manual bookkeeping needs explicit audit at each commit.

## Task status

- Task 1: in progress; video tests written before implementation.
- Tasks 2-7: pending.

Task 1 evidence: 5 video tests RED -> GREEN; official source independently decodes 70 frames at 960x540, 24 FPS, SHA-256 7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8. First three previews saved. CUDA verification awaits wheel download.
Task 2 evidence: 11 cache/model-contract tests observed failing before implementation; implementation now checked by full suite. Official source revisions recorded in manifests/sources.json.

Task 3 code verified: exact course metrics, HDF5 axes, split indices, PFM endianness, stereo conversion, and 15-pair guard tested. Real stereo run: mean pair MAE 1.894695421 px, mean coverage 0.756890603. PFM display-scale regression observed RED then GREEN; the raw pixel samples are preserved. NYU neural evaluation awaits data/checkpoints.
Ruling: Official Middlebury ZIP uses trainingQ/ rather than the PDF shorthand trainQ/. Use its 15 supplied trainingQ pairs. Cost if wrong: pairing/count/hash guards prevent a silent substitution.
Task 4 code verified: 8 identity/translation/bilinear/border/flow-direction checks RED -> GREEN. Full suite now 45 tests. Visual diagnostics pending baseline inference.
Task 5 code verified: 8 numerical/scene-cut/causality/flat-map/synthetic checks RED -> GREEN; real-video tuning is pending.
