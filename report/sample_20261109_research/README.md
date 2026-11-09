# Research-ordered Project 3 sample report

This separate version organizes the same verified experiments by research question. The original and extended reports remain unchanged. No experiment is rerun, no candidate is newly selected, and no numerical result is replaced.

## Reading order

1. Introduction: questions and scope.
2. Related Work: geometric, monocular and video depth methods.
3. Method and Evaluation Setup: datasets and splits, reference methods, our flow-guided fusion, metrics and evidence limits.
4. Experiments:
   - 4.1 Stereo accuracy, completeness and controlled LR rejection (Part 1).
   - 4.2 UniDepth metric accuracy, scale and boundary failures (Part 2).
   - 4.3 Frame-wise DA, offline VDA and streaming quality/cost (Parts 1–2).
   - 4.4 Our fusion: E0–E4, history/confidence/strength controls, then the bounded refinement (Part 3).
   - 4.5 Background blur: separate threshold changes from DA/E4 changes.
5. Discussion and Conclusion: observed failures, limitations and explicitly untested next steps.

Course Part labels remain visible. The independent Middlebury and NYUv2 evaluations are not presented as a common-dataset model ranking. VDA is an external video reference, not an input to our DA fusion.

## Evidence and versioning

`data/`, `extension/`, `images/`, bibliography, CVPR style and asset receipts are byte-identical to `../sample_20261109_extension/`. The original 32 numerical table rows and 16 completed review records are preserved. Not every supporting image is placed in the eight-page body.

- `main.tex`: editable report.
- `revision_manifest.json`: new structure, parent receipt and scope.
- `inherited_report_provenance.json`: exact parent validation receipt; its PDF and archive hashes refer to the earlier extended version.
- `report_provenance.json`: this revision's source, numerical, PDF and visual verification receipt.

The new PDF is stored locally at `outputs/submission_20261109_research/MMDEVC_Project3_Research_Report.pdf`. It has eight body pages plus one references page. Large outputs are Git-ignored.

The previous `outputs/submission_20261109_extension/depth_results_extended.zip` is preserved unchanged and still contains the earlier extended report. This report-only revision does not replace that video/evidence package.

## Build and verify

From this report directory, create the output directory and run:

```text
tectonic -X compile --only-cached --untrusted --keep-logs --keep-intermediates --outdir ../../outputs/submission_20261109_research/report_build main.tex
```

Copy `report_build/main.pdf` to the named output, inspect all pages and check the course page limit. The retained validation receipts describe the report at generation time. Report-only verification utilities are archived in Git revision `d55270f`; they are not part of experiment reproduction.

Compilation uses the files in this directory. Experiment statistics and figures can be regenerated with the per-Part commands in [the analysis guide](../../docs/analysis_workflow.md).

The 70/100-frame source discrepancy, absent video depth GT, post-hoc status and sampled visual-review scope remain explicit. Original E4 is retained; the bounded candidates did not establish clear visual improvement. The authors still need to review and accept this sample before course submission.

The report receipts preserve the original generation hashes, including old tooling and README bytes. They are historical records, not a claim that the current documentation has those same hashes. No experiment result or PDF was changed during the tooling cleanup.
