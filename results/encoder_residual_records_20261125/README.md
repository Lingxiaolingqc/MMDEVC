# Encoder-feature residual study

Fixed queue: ES0 -> EF2 -> EF3 -> ESS1 -> EP2 -> EP3; 577 fitting / 218 validation, three seeds.
No test, all-795 refit, LoRA or sample-efficiency runs. See docs/encoder_residual_plan.md for adaptations.

Completed: ES0
Pending: EF2, EF3, ESS1, EP2, EP3

| Method | AbsRel mean +/- SD | RMSE | delta1 | Boundary F1 | Band AbsRel | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| ES0 | 0.104583 +/- 0.000847 | 0.440657 | 0.882852 | 0.465408 | 0.162597 | 30737 |
| S0 | 0.099848 +/- 0.000875 | 0.436360 | 0.919802 | 0.462688 | 0.163552 | 28689 |
| F3 | 0.094364 +/- 0.000835 | 0.413314 | 0.925772 | 0.452682 | 0.154347 | 27089 |

Current selection: F3 (provisional; queue incomplete)

Best/final weights and feature arrays remain local under outputs/part3_improvement/encoder_residual_20261125.
Full per-image metrics, curves, seed variation, paired scene CIs and best-to-final trajectories are in analysis/review.json.
Read analysis/interpretation.md and analysis/qualitative_review.json for inspected S0/ES0 cases; figures are in analysis/visualizations/.
Numerical source is preserved in numerical_source.zip. Historical decoder records are reused after reference_audit.json checks.
S0 init uses the first decoder input branch (encoder block3 under official last stacking), then camera conditioning/to_latents; ES0 uses block12. This changes layer depth too, not just decoder conditioning.
Costs recorded during different training runs are descriptive, not an isolated latency benchmark.

## Run / resume

Requires the existing development prediction cache, audited decoder reference runs and the verified
StyleSketch disk_inputs cache (RGB/log-depth, GT, masks only). See config paths; these are reused read-only.

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py prepare
& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py preflight
& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py queue --through ES0 --resume
# Continue the registered remaining methods in order:
& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py queue --resume
# Analyze completed three-seed comparisons; run rendering while training is stopped:
& $py -X utf8 scripts/part3_improvement/render_encoder_residual.py
& $py -X utf8 scripts/part3_improvement/analyze_encoder_residual.py --export
```

Completed runs are verified and reused. A failure stops the queue; --resume restores optimizer/RNG state.
Do not run the standalone benchmark concurrently with training.
