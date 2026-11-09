# Extended Project 3 sample report

This separate 2026-11-09 version adds a matched official VDA offline/streaming comparison and one bounded confidence-relaxation experiment. The original report in `../sample_20261109/` remains unchanged.

The report has eight body pages and one references page. Original E4 remains the selected main method: the new candidates produce small temporal-diagnostic reductions without a clear sampled visual benefit. Streaming lowers peak allocated GPU memory but introduces startup variation; its measured per-frame latency does not meet 24 fps. No foundation-model training or metric-video experiment is claimed.

## Contents

- `main.tex`, official CVPR style, bibliography: editable sample report.
- `data/`, `asset_provenance.json`: exact copies of the original compact evidence and figure receipts.
- `extension/`: new mode timing, regional diagnosis, frozen dev decision, controlled comparison and independent reviews.
- `images/`: all preserved original figures plus new matched figures. Some detailed figures are retained as supporting evidence rather than placed in the eight-page body.
- `extension_assets.json`: hashes linking the new compact evidence and figures to original local arrays and media.
- `report_provenance.json`: numeric table, PDF, font, visual review and ZIP validation receipt.

## Build

From this directory, using the existing LaTeX environment:

```text
tectonic -X compile --only-cached --untrusted --keep-logs --keep-intermediates --outdir ../../outputs/submission_20261109_extension/report_build main.tex
```

The inspected deliverables are stored locally in `outputs/submission_20261109_extension/`: `MMDEVC_Project3_Sample_Report_Extended.pdf` and `depth_results_extended.zip`. Large outputs are Git-ignored. The ZIP includes the previous main experiment payloads plus the new report and supplement. Full NPY arrays, datasets and model weights remain local. The private acceptance guide is excluded.


## Interpretation

The 70-frame course source differs from the requested first 100 frames; all available frames are retained without duplication. User acceptance of this handling is not instructor approval. Video has no depth GT. Temporal diagnostics, blur strength, and output changes do not measure depth accuracy. Streaming display normalization uses the full clip only for offline review. Figures represent explicitly recorded samples and consecutive windows, not full real-time playback. Existing main and new repeated timings have distinct measurement protocols.

The targeted refinement was selected on Tokyo before applying unchanged parameters to the already-viewed course clip. The course comparison is post-hoc. The original E4 is retained because neither candidate establishes clear visual benefit. The authors must review the report and follow their course's attribution/submission requirements; generating this sample is not submission or acceptance.

The report receipts preserve the original generation hashes, including old tooling and README bytes. They are historical records, not a claim that the current documentation has those same hashes. No experiment result or PDF was changed during the tooling cleanup.
