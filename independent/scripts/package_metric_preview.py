"""Package the required videos with the metric improvement evidence."""
import argparse
import json
from pathlib import Path

from depth_project.submission import build_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/depth_results_metric_preview.zip'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    relative_paths = [
        'artifacts/metric-refiner.safetensors',
        'configs/metric-final.json',
        'results/metric-improvement.json',
        'results/metric-baseline-per-image.csv',
        'results/metric-residual_refiner-per-image.csv',
        'results/metric-figure-selection.json',
        'results/figures/metric-comparison.png',
        'results/figures/metric-application.png',
        'results/figures/metric-ablation.png',
    ]
    inventory = build_submission(
        root / 'runs/final', root / args.output,
        allow_short=True, report_path=root / 'report/independent-report.pdf',
        supplemental_files={path: root / path for path in relative_paths},
    )
    print(json.dumps({'submission_ready': inventory['submission_ready'], 'files': len(inventory['files']) + 1, 'frames': inventory['processed_frames'], 'warning': inventory['warning']}, indent=2))


if __name__ == '__main__':
    main()
