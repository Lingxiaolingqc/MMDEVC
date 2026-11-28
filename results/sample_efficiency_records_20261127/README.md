# Sample-efficiency study, 2026-11-27

Protocol and execution order: ../../docs/sample_efficiency_plan_20261127.md.

All 72 development training stages and all 145 frozen test checkpoint evaluations are complete. Each checkpoint has 654 official test-image records. The cached/live validation bridge passed; all 24 method/count groups have three seeds, and every predeclared best/final state is retained. Numerical analysis and actual fixed-image review are complete. Start with [the final report](test_analysis/final_report.md), [structured conclusions](test_analysis/final_review.json), and [completion evidence](test_analysis/completion_audit.json).

R: decoder F3. L: encoder last-four-block rank-8 LoRA. J: joint LoRA plus F3. S: F3 trained after freezing the corresponding L. Three seeds at 100/200/300/400/500/577 images; zero-shot B0 is shared.

Official CUDA antialiased bilinear backward is nondeterministic. Checkpoint parameter restoration was exact; next-update replay is checked against the documented tolerance. Preflight failures are preserved; only the final passing preflight authorizes formal training. No old experiment source was changed.

Final test completion: test_complete.json. The earlier queue.json, progress records and recovery snapshots retain their at-time state; they are historical evidence. Large caches, images and best/final/resume weights remain local under outputs/part3_improvement/sample_efficiency_20261127/. Test visual_review.json links 72 final panels to actually inspected snapshots; two additional endpoint failure cases and earlier views are preserved. Independent checks recomputed five metrics on all 435 saved fixed predictions.

From the repository root, use the CUDA Python environment. The training and development-analysis commands below describe reproduction before test freezing, not commands to rerun over this frozen study:

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/part3_improvement/run_sample_efficiency.py queue
& $py -X utf8 scripts/part3_improvement/analyze_sample_efficiency.py --render --export-records
```

The queue resumes saved jobs and refuses a second concurrent worker. Analysis reads completed outputs without running inference. per_seed.csv separates raw best, stopping final, operational stability, target steps, boundary scores and costs. Scene-bootstrap intervals and training-seed variation are reported separately. R uses precomputed foundation features; its training time excludes that shared extraction cost. Stage costs include the corresponding L's full training, not only its best iteration.

After all 72 development stages and actual visual review, evaluate_sample_efficiency.py provides bridge → freeze → queue. This study has completed the bridge, freeze and all 145 test identities. The bridge reproduced all 218 validation predictions exactly under the declared cached/live interface. The frozen test list contains B0 plus best/final for every predeclared run; it does not use test results to select a candidate. The official test was examined in prior project experiments, so it cannot be described as a previously unseen blind test.

For the existing frozen study, analyze_sample_efficiency.py --test --render --export-records builds separate test tables and fixed first/middle/last test examples without inference. Keep --test: rerunning development analysis would rewrite artifacts referenced by test_frozen.json. In these tables, best always means the checkpoint selected on validation, even when final happens to have a better test score. Stable membership comes from validation only; conditional stable-final scores always include their seed counts and remain null if no seed qualifies.

Visual review records explicitly name the inspected files and scope; rendering a figure alone does not count as review. Negative examples, post-best deterioration and failed attempts are retained.

Two formal training file-access failures and one test progress-file access failure were recovered without changing batch, configuration or checkpoint identities. Failure traces and replay costs remain in the archive. The final audit verified that 504 protected training artifacts stayed unchanged. Early stopping completed all runs, but only 31/72 met the predefined stability rule; stopped and stable are distinct.
