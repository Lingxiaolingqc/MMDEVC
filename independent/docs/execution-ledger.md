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

Task 1: complete except external 100-frame compliance issue. CUDA 12.8 / torch 2.8.0 / torchvision 0.23.0 verified on RTX 4070 with tensor-square result [4.0, 9.0]; 5 video checks pass. Learning note and first-three-frame previews saved; no claim of personal mastery made.
Task 2: DA V2-Small full inference complete on development Tokyo first 100 frames and held-out rollercoaster all 70 real frames, at 640x360 output, official input_size 518 FP16. VDA-Small offline 518 FP16 inference in progress. Official DA checkpoint SHA-256 715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378 matches the published Hugging Face file page.

Task 2: complete. Our own DA V2-Small 518 FP16 outputs exist for Tokyo/rollercoaster; official VDA-Small offline outputs exist at input_size 518 and 392. All clips preserve true frame IDs, 640x360 output, original FPS, raw float caches, and explicit short-source status.
Ruling: Use VDA input_size 392 as the main reference because the default 518 run reported 8014 MiB allocated and 11858 MiB reserved against 8188 MiB physical VRAM. The 392 run used 3072 MiB allocated / 4194 MiB reserved and 6.941 s for 70 frames; preserve the 518 run as a resource sensitivity check. Cost if wrong: VDA visual detail may decrease; report both settings and keep each method/improvement pair fixed.
Task 3: complete. All 15 stereo pairs and all 654 NYU test images independently evaluated. NYU AbsRel=0.1145457596, RMSE=0.3988876971 m, delta1=0.8922178330, raw metric scale, exact course crop, no fitting. Peak allocated 485.84 MiB; median 58.37 ms/image.
Task 6 development: baseline temporal proxy 0.010318; 12 adaptive candidates plus fixed ablations evaluated only on Tokyo first 100 frames. Selected adaptive_w0.75_b0.5_d0.5: proxy 0.005926, edge-retention proxy 0.947, synthetic settling 1 frame, synthetic MSE 0.07988 versus 0.09080 raw. Final configuration is now being committed before held-out fusion evaluation. No held-out fusion score has been inspected.

Task 6 final evaluation: frozen configuration commit f35e63ce487e208792d621c570f9c28f7a243499 predates the held-out fusion run. Rollercoaster baseline proxy 0.107973; selected adaptive 0.088292, reduction 18.2278068%; edge retention 0.976. Occlusion-only has a smaller proxy but edge retention 0.794; EMA is worse under motion alignment. Final configuration remains unchanged.
Task 7 in progress: synchronized comparison and background-blur videos generated for all 70 true frames at 24 FPS. Scientific figures, runtime repeats, report, and package verification remain.

Ruling: After profiling the original frozen pipeline, add a separately labeled engineering study of flow resolution on development data only. The initial held-out result remains the primary result; do not present the follow-up as a fresh untouched test. Cost if wrong: another engineering round weakens confirmatory independence, so preserve both configurations/results and disclose chronology.
Engineering selection: development flow width 320 had temporal proxy 0.006119732 (within 5% of original 0.005926), edge retention 0.9493, flow computation 2.756 s for 99 pairs. Width 480 scored 0.006014106; select the smallest eligible width 320 by the declared development rule. Efficient configuration is frozen before secondary held-out evaluation.
