# Resolution sensitivity

Both controlled runs are complete: fixed 577/218 split, seed 320112, unchanged architecture and 80-epoch training/selection protocol; only spatial refiner resolution changes.

| Method | Resolution | AbsRel | Boundary macro F1 | Selected / trained epochs | Peak training allocated MiB | Run elapsed s |
|---|---|---:|---:|---:|---:|---:|
| R0 | 240x320 | 0.11321818 | 0.46072419 | 2 / 24 | 362.95 | 86.11 |
| R0 | 480x640 | 0.11321093 | 0.46014348 | 2 / 24 | 1337.75 | 226.20 |
| Internal-feature CNN | 240x320 | 0.10064991 | 0.46255194 | 76 / 80 | 436.34 | 298.63 |
| Internal-feature CNN | 480x640 | 0.10071245 | 0.46325813 | 76 / 80 | 1579.96 | 1014.27 |

For the internal-feature CNN, 480 minus 240 changes AbsRel by +0.00006254 (scene-cluster 95% interval [-0.00004072,+0.00016565]) and F1 by +0.00070620 ([-0.00022249,+0.00185210]). Neither establishes an improvement. R0 accuracy is almost unchanged; its small F1 decline and boundary-band error reduction answer different questions and are not evidence for new contours. This is a one-seed sensitivity check, not training-seed uncertainty.

The already-recorded promotion rule required at least 1% relative lower AbsRel AND +0.002 absolute F1 for the selected method before adding two further seeds. Both requirements fail. Keep240, without expanding resolution search. Training peak allocated memory grows about3.62x and run elapsed time about3.40x for the selected method; these are recorded training runs, not a controlled deployment latency benchmark. Foundation resolution and native internal-feature grid are unchanged.

Visual inspection: compared fixed validation images0005 and1440 for the internal-featureCNN at both resolutions under identical0–10m depth and±0.5m correction limits. Broad corrected depth layouts remain very similar. Higher resolution produces finer texture in the correction map without visible recovery of the missing fine contours in these overviews. This limited inspection does not claim all218 images are visually unchanged. No new test data was used.

Full per-image scores, paired distributions, bootstrap details and comparison.png/pdf: outputs/part3_improvement/residual_study_20261118/analysis/resolution/. Decision and hashes: resolution_selection.json in the same study root.
