"""Render a six-FPS diagnostic with a fixed-scale prediction difference panel."""
import json
from pathlib import Path

import cv2
import numpy as np

from depth_project.cache import load_predictions
from depth_project.render import normalize_depth, render_comparison
from depth_project.video import inspect_video, iter_rgb_frames


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'runs/final'
    summary = json.loads((run / 'summary.json').read_text())
    baseline, baseline_metadata = load_predictions(root / 'runs/baselines/rollercoaster-dav2.npz')
    fused, fused_metadata = load_predictions(run / 'adaptive.npz')
    assert baseline_metadata['frame_ids'] == fused_metadata['frame_ids']
    assert baseline_metadata['source_sha256'] == fused_metadata['source_sha256']
    display_range = summary['display_ranges']['DA V2 Small (relative)']
    baseline_display = normalize_depth(baseline, display_range)
    fused_display = normalize_depth(fused, display_range)
    difference = np.abs(fused_display - baseline_display)
    source = root / 'third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4'
    height, width = baseline.shape[1:]
    rgb = np.stack([
        cv2.resize(frame.rgb, (width, height), interpolation=cv2.INTER_AREA)
        for frame in iter_rgb_frames(source, len(baseline))
    ])
    output = root / 'runs/diagnostics/fusion-difference-slow.mp4'
    methods = {
        'RGB | quarter-speed diagnostic': rgb,
        'DA V2 | fixed original display range': baseline,
        'Fusion | same display range': fused,
        'Absolute change | bright>=2% display range': difference,
    }
    ranges = {name: display_range for name in methods}
    ranges[list(methods)[-1]] = (0, 0.02)
    render_comparison(baseline_metadata['frame_ids'], methods, ranges, 6, output)
    info = inspect_video(output, len(baseline))
    assert info.decoded_frames == len(baseline) and info.fps == 6
    record = {
        'frame_count': len(baseline),
        'original_fps': summary['fps'],
        'diagnostic_fps': 6,
        'prediction_display_range': display_range,
        'difference_display_range': [0, 0.02],
        'mean_absolute_change_fraction_of_display_range': float(difference.mean()),
        'median_absolute_change_fraction_of_display_range': float(np.median(difference)),
        'p95_absolute_change_fraction_of_display_range': float(np.percentile(difference, 95)),
        'pixels_changing_over_one_percent_of_display_range_percent': float(100 * np.mean(difference > 0.01)),
        'first_frame_max_change': float(difference[0].max()),
        'preview_frame_selection': 'largest frame-mean absolute prediction change',
        'preview_frame_index': int(np.argmax(difference.mean(axis=(1, 2)))),
        'interpretation': 'Measures changed predictions. Accuracy requires ground-truth depth. Difference colors saturate at 2% of the original prediction display range.',
        'video': str(output.relative_to(root)).replace('\\', '/'),
    }
    (root / 'results/fusion-difference-diagnostic.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    preview = output.with_suffix('.png')
    capture = cv2.VideoCapture(str(output))
    capture.set(cv2.CAP_PROP_POS_FRAMES, record['preview_frame_index'])
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert cv2.imwrite(str(preview), frame)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
