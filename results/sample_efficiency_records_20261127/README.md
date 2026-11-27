# Sample-efficiency study, 2026-11-27

Protocol and execution order: ../../docs/sample_efficiency_plan_20261127.md.

This archive contains the frozen 72-stage plan, nested data subsets, passed engineering checks, the shared 218-image validation baseline, and records for completed runs. See manifest.json and analysis/summary.json for the snapshot's completion count. Training is in progress; incomplete method/size/seed cells are not results. No new official test evaluation has been run for this study.

R: decoder F3. L: encoder last-four-block rank-8 LoRA. J: joint LoRA plus F3. S: F3 trained after freezing the corresponding L. Three seeds at 100/200/300/400/500/577 images; zero-shot B0 is shared.

Official CUDA antialiased bilinear backward is nondeterministic. Checkpoint parameter restoration was exact; next-update replay is checked against the documented tolerance. Preflight failures are preserved; only the final passing preflight authorizes formal training. No old experiment source was changed.

Live local progress: outputs/part3_improvement/sample_efficiency_20261127/queue.json. Large caches and weights remain local.

From the repository root, use the CUDA Python environment:

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/part3_improvement/run_sample_efficiency.py queue
& $py -X utf8 scripts/part3_improvement/analyze_sample_efficiency.py --render --export-records
```

The queue resumes saved jobs and refuses a second concurrent worker. Analysis reads completed outputs without running inference. per_seed.csv separates raw best, stopping final, operational stability, target steps, boundary scores and costs. Scene-bootstrap intervals and training-seed variation are reported separately. R uses precomputed foundation features; its training time excludes that shared extraction cost. Stage costs include the corresponding L's full training, not only its best iteration.

After all 72 development stages and actual visual review, evaluate_sample_efficiency.py provides bridge → freeze → queue. The bridge must confirm cached/live inference parity on validation before test. The final test list contains B0 plus best/final for every predeclared run; it does not use test results to select a candidate. This later GPU path has not yet been runtime-validated or executed. The official test was examined in prior project experiments, so it cannot be described as a previously unseen blind test.

Visual review records explicitly name the inspected files and scope; rendering a figure alone does not count as review. Negative examples, post-best deterioration and failed attempts are retained.
