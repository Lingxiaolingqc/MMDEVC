# Independent Computer Vision Project 3

An independently developed, local-first implementation of the revised AIAA3201 assignment, with Direction A: flow-guided temporal depth fusion.

**Current status:** planning and isolated Git setup. There are no new performance results yet.

This checkout lives beside `MMDEVC`, on `feat/independent-direction-a`. It shares the GitHub repository's history for eventual integration, but only `independent/` is checked out. No groupmate implementation or historical result is used as our implementation or baseline.

## Start here

- [Proposed design and exact assignment requirements](docs/superpowers/specs/2026-11-08-direction-a-design.md)
- [Implementation and experiment plan](docs/superpowers/plans/2026-11-08-direction-a.md)

## Local machine and compute

The detected GPU is an NVIDIA GeForce RTX 4070 Laptop GPU with 8,188 MiB VRAM. The revised instructions require Small depth models and recommend the video repository's default FP16. The course GPU is optional; the assignment specifies no bonus for using it. Use it only if a measured memory or runtime limit makes it useful. Course data can also be downloaded from the official links; server access is not required to begin.

## Safe Git workflow

Run Git commands from `CV_Project3_Independent`, not from the parent Project folder or `MMDEVC`.

```powershell
git branch --show-current
git status --short
git add -- independent/<specific-file>
git diff --cached --stat
git commit -m "feat: describe one independent contribution"
# When publishing is intended, explicitly name this branch:
git push -u origin HEAD:refs/heads/feat/independent-direction-a
```

Local hooks reject commits outside the independent branch prefixes, staged changes outside `independent/`, and pushes to other remote branches, including `main`. Hook enforcement is local and can be bypassed; it does not install GitHub branch protection. The initial setup enables these hooks only in this checkout. It does not alter global Git configuration.

Future feature branches should use `feat/independent-*`, `fix/independent-*`, `docs/independent-*`, or `experiment/independent-*`. Use pull requests for integration; do not push to `main` or overwrite the group's existing paths.

Large inputs, checkpoints, upstream repositories, prediction caches, and rendered videos stay ignored. Track small manifests, source, configurations, reproducible summaries, and selected report figures when implemented.

## Independence and learning

Our own runs will establish the baselines. Each stage follows: worked example -> explanation -> implementation -> meaningful checks -> a short explanation in the user's own words. Official model components will be attributed and pinned; independently written glue, fusion, evaluation, and analysis will be identified clearly.
