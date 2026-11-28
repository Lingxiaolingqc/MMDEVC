# Sample-efficiency study, 2026-11-27

Protocol and execution order: ../../docs/sample_efficiency_plan_20261127.md.

All 72 development training stages and their numerical and actual fixed-image reviews are complete. The cached/live validation bridge passed, and the 145 test checkpoint identities were frozen before this study's test inference. Official 654-image test evaluation is in progress. See test_analysis/summary.json for this archive snapshot's completed jobs and pending list; incomplete method/size/seed cells are not final results. Do not treat the development completion count as completion of test evaluation or final delivery.

R: decoder F3. L: encoder last-four-block rank-8 LoRA. J: joint LoRA plus F3. S: F3 trained after freezing the corresponding L. Three seeds at 100/200/300/400/500/577 images; zero-shot B0 is shared.

Official CUDA antialiased bilinear backward is nondeterministic. Checkpoint parameter restoration was exact; next-update replay is checked against the documented tolerance. Preflight failures are preserved; only the final passing preflight authorizes formal training. No old experiment source was changed.

Live local test progress: outputs/part3_improvement/sample_efficiency_20261127/test_progress.json; individual checkpoint progress is in final_test/. The earlier queue.json records the completed training queue. Large caches, images and weights remain local.

From the repository root, use the CUDA Python environment. The training and development-analysis commands below describe reproduction before test freezing, not commands to rerun over this frozen study:

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/part3_improvement/run_sample_efficiency.py queue
& $py -X utf8 scripts/part3_improvement/analyze_sample_efficiency.py --render --export-records
```

The queue resumes saved jobs and refuses a second concurrent worker. Analysis reads completed outputs without running inference. per_seed.csv separates raw best, stopping final, operational stability, target steps, boundary scores and costs. Scene-bootstrap intervals and training-seed variation are reported separately. R uses precomputed foundation features; its training time excludes that shared extraction cost. Stage costs include the corresponding L's full training, not only its best iteration.

After all 72 development stages and actual visual review, evaluate_sample_efficiency.py provides bridge → freeze → queue. This study has completed the bridge and freeze and is running that test queue. The bridge reproduced all 218 validation predictions exactly under the declared cached/live interface. The frozen test list contains B0 plus best/final for every predeclared run; it does not use test results to select a candidate. The official test was examined in prior project experiments, so it cannot be described as a previously unseen blind test.

For the existing frozen study, analyze_sample_efficiency.py --test --render --export-records builds separate test tables and fixed first/middle/last test examples without inference. Keep --test: rerunning development analysis would rewrite artifacts referenced by test_frozen.json. In these tables, best always means the checkpoint selected on validation, even when final happens to have a better test score. Stable membership comes from validation only; conditional stable-final scores always include their seed counts and remain null if no seed qualifies.

Visual review records explicitly name the inspected files and scope; rendering a figure alone does not count as review. Negative examples, post-best deterioration and failed attempts are retained.
