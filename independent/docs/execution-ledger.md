# Execution ledger - plan: docs/superpowers/plans/2026-11-08-direction-a.md

## Objective revision on 2026-11-11

## Metric execution on 2026-11-12

User authorized strongest course-compliant performance search. Implementation plan docs/superpowers/plans/2026-11-12-metric-performance.md executes the approved performance-search design. Ruling: reuse this isolated feature checkout and execute inline, with native Windows bookkeeping. Previous bash bookkeeping incompatibility remains applicable. Ruling: keep foundation input resolution fixed and exclude multi-resolution candidates from this comparison to respect the course resource rule.

All 795 NYU development images have original and horizontal-flip predictions. Whole scene split using seed 320112 gives 577 fitting images and 218 validation images in 62 reserved scenes, with no test overlap. Exact identities and IDs are saved. Five numerical calibration/filter tests and three residual-refiner identity/finite/multiplicative tests were observed RED then GREEN. CLI integration regression was observed RED then GREEN. Complete suite 86 passing.

The 28 simple candidates select flip weight 0.25 and global scale 0.9413383. Validation AbsRel falls from 0.1206857 to 0.1082135. Guided filtering and additional flip weights remain recorded as negative/alternative findings. A 388,001-parameter RGB-guided residual CNN trained for 50 epochs before the declared 20-epoch patience rule stopped training. Best validation checkpoint is epoch 30, AbsRel 0.1027521, 14.86% below the validation baseline. Only fitting images supplied gradients; full-resolution validation selected the checkpoint. The foundation checkpoint remained frozen. This is supervised NYU development adaptation.

Ruling: distribute the small independently trained refiner as safetensors under artifacts/ so reproduction does not depend on ignored run caches or unsafe pickle loading. Foundation checkpoints remain external. The method configuration and refiner digest are being committed before the 654-image test evaluation. Earlier baseline/failure examples have already been examined, and this history will be disclosed. No improved test score has yet been inspected.

The user prioritizes strongest measurable performance, with freedom to change direction. docs/performance-search.md supersedes the initial fixed Direction A choice and blanket exclusion of adaptation. Main quantitative target is improvement over our unchanged UniDepthV2-S NYU reference under the teacher's exact protocol. A named SOTA comparison requires a matched rerun. New accuracy experiments remain proposed.

The user observed no visible demo difference. Direct comparison of all 70 saved maps found mean absolute change of 0.2215% of the frozen display range, median zero, 95th percentile 1.6329%, and 8.9145% of pixels changing by more than 1% of that range. These are prediction differences, with unknown video depth accuracy. A separate six-FPS diagnostic retains all frames and adds a fixed 0-2%-range absolute-change display. Its 70 decoded frames and rate were verified. The preview is selected by maximum mean frame change, and its layout was visually inspected. The main video and preview archive retain their original measured outputs.

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

- Tasks 1-7: implemented and verified; see the chronological evidence below and final artifact checks.
- External submission dependency: obtain 100 genuine source frames or TA clarification. Current exports remain explicitly provisional at 70 frames.
- Teaching notes are prepared, but the user's understanding has not been assessed in an interactive walkthrough.

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

Task 6: complete. Main frozen study: 18.23% temporal proxy reduction with 97.58% edge retention. Secondary engineering study: 16.82% reduction with 97.64% retention; flow width 320 selected on development data only. Full uncached pipeline median FPS: 2.7691 original versus 4.3442 efficient, a 56.9% speed increase across three synchronized trials per setting. Results are separate and chronology is disclosed.
Task 7 code verification: fixed whole-clip rendering, real frame count/FPS, blur behavior, and provisional-archive enforcement tests pass. Report/visual/archive checks are still in progress.
Reproducibility note: early runs predated automatic start-of-run Git/source hashing. They retain full input/checkpoint/upstream/config identities; no source-snapshot hash is retrospectively invented. New runs now automatically capture Git SHA, dirty state, source/config hashes, environment, and lockfile hash at start.

Final fresh code review: gpt-6-astra reviewer independently ran 67/67 pre-fix tests. No Critical findings. Two Important findings were reproduced and fixed in one pass.
Final: fixed false 100-frame compliance when a caller requested a shorter smoke-test limit -- 2 end-to-end compliance regressions RED -> GREEN. Course readiness now derives from actual decoded first-100 frame IDs/count, independently of caller limit; messages report actual counts.
Final: fixed acceptance of unrelated raw evidence or wrong rendered dimensions in packaging -- 8 source/frame/shape/kind/key/config/dimension regressions RED -> GREEN. Raw baseline/reference/fusion relationships and explicit video layouts now validate before archive creation. Full suite: 77/77 passing. Existing default-limit numerical results were not affected; only metadata was explicitly enriched from observed source IDs/count.
Final: minor (deferred): automatic verification that the loaded official checkout is clean and matches the manifest revision. Current source HEADs match the recorded upstream identities; future manual upstream edits must trigger new provenance/cache identities.
Final: Ruling: the reviewer did not judge report/README/figure authoring because it was still underway. Parent will render and inspect the PDF and videos. Cost if wrong: document presentation errors require the separate visual QA to catch them.
Final: Ruling: obtaining 100 source frames is external, not a code defect. Keep strict readiness false and preserve the provisional warning pending course clarification. Cost if wrong: a submission could otherwise claim an unmet requirement.
Final: Ruling: the reviewer did not repeat neural experiments; measured performance remains supported by this session and saved per-image/per-frame evidence, rather than an independent GPU rerun. Cost if wrong: an undiscovered model/evaluation integration error would need a separate reproduction.
Final: Ruling: RAFT and affine alignment remain optional and unused. They are not needed for the implemented controlled Direction A study. Cost if wrong: potential gains from other backends remain unmeasured.

Task 7 final evidence (2026-11-09): the official CVPR report compiled successfully offline with no LaTeX warnings: seven body pages plus one reference page, 13 resolved references. Every rendered page was visually inspected. Actual scientific figures include static comparisons, fixed video frames, the three highest-AbsRel NYU failures, and development flow/history/support/weight diagnostics. The diagnostic figure was added during final QA, after fusion implementation; geometry and synthetic correctness had already been checked numerically before real-video fusion.

Final artifact validation: all four primary/efficient videos decode 70 frames at 24 FPS; comparison canvas 1280x776, application canvas 1280x388. Frames 0, 35, and 69 from each export were visually inspected. Three raw caches contain finite float32 arrays shaped 70x360x640, with matching source/frame/config relationships. The 69,415,126-byte preview ZIP contains nine entries including its manifest; all archived file sizes, SHA-256 values, and CRC checks pass. It includes the PDF and raw predictions. Strict packaging was exercised against the real run and rejected the short source without creating depth_results.zip. Full evidence: results/deliverable-verification.json.

Final validation: 77/77 tests pass in the complete suite. Local document links resolve. No weights, datasets, raw caches, videos, or ZIPs are staged. Report author Eugene remains a draft to confirm before group submission. Keep the checkout and feature branch; the user has already requested branch-based updates. Publish only that named branch, with no merge or force push and no unsolicited PR creation.

Isolation audit: sibling MMDEVC remains clean on main at 6fccdfdb4759b2b8e4391926843105892e8986e4. The independent branch diff against its base contains only independent/ paths. Vendored official ieeenat_fullname.bst retains one upstream trailing space in its header; leave the attributed file unchanged and exclude that file from our whitespace check.
