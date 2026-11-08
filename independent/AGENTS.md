# Independent AIAA3201 Project 3

## Scope and ownership

- This is the user's independent implementation. Use the revised instructor PDF, official model code/checkpoints, original papers, official datasets, and code developed here.
- Do not inspect, adapt, or import the groupmate's implementation, configurations, results, or selected parameters. Git ancestry exists solely for a clean future merge.
- All tracked contributions belong under `independent/`. Do not edit paths belonging to the group's existing implementation.
- The enclosing Obsidian vault and the sibling `MMDEVC` checkout are separate repositories. Never stage, commit, switch, pull, reset, or clean them as part of this work.

## Branch workflow

- Work on `feat/independent-*`, `fix/independent-*`, `docs/independent-*`, or `experiment/independent-*` branches.
- Never commit or push to `main` or `master`. Never force-push shared history or merge without the user's instruction.
- Confirm the repository root, branch, and diff before each commit. Stage explicit paths under `independent/`, never the parent vault.
- Keep the branch based on shared Git history so eventual integration adds this folder instead of replacing existing files.
- Local hooks in `.githooks/` enforce branch restrictions when `core.hooksPath` points to `independent/.githooks`.

## Research and evaluation

- The authoritative task source is `../Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf` relative to the checkout root. The older PDF with spaces in its name is superseded.
- Follow the revised Small-model rule. Start locally on the RTX 4070 Laptop GPU, 8 GB VRAM. Server use is optional and provides no documented assessment bonus.
- Save raw float predictions, input hashes, checkpoint/source revisions, configuration, hardware, precision, and timing scope.
- Keep development separate from final evaluation. Freeze configuration before final runs. Do not select parameters using the groupmate's results.
- NYU metrics use raw metric depth without test-set alignment. Frame-wise/video relative predictions must not be described as meters.
- Video temporal scores, RGB-edge alignment, and stability are diagnostics, not proof of metric-depth accuracy. Never hide flicker with per-frame display normalization.

## Teaching

Start each learning stage with one concrete example and a familiar analogy. Trace the numbers before introducing notation. Explain inputs, pretrained parameters, predictions, and ground truth separately. Use a small diagram when it clarifies the mechanism. Build a testable stage, then record what the user can explain before moving on.

## Current phase

The user approved implementation with "go" on 2026-11-08. Execute the design and implementation plan in `docs/superpowers/`, maintaining the execution ledger and reporting actual verification evidence.
