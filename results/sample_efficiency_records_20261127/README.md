# Sample-efficiency study, 2026-11-27

Protocol and execution order: ../../docs/sample_efficiency_plan_20261127.md.

This initial archive contains the frozen 72-stage plan, nested data subsets, passed engineering checks and the shared 218-image validation baseline. Formal training has started; this archive is not evidence of completed training or test evaluation.

R: decoder F3. L: encoder last-four-block rank-8 LoRA. J: joint LoRA plus F3. S: F3 trained after freezing the corresponding L. Three seeds at 100/200/300/400/500/577 images; zero-shot B0 is shared.

Official CUDA antialiased bilinear backward is nondeterministic. Checkpoint parameter restoration was exact; next-update replay is checked against the documented tolerance. Preflight failures are preserved; only the final passing preflight authorizes formal training. No old experiment source was changed.

Live local progress: outputs/part3_improvement/sample_efficiency_20261127/queue.json. Large caches and weights remain local.
