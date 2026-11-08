"""Analyze raw DA relative video output, with optional post-hoc flow diagnostics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from _common import ROOT, repo_path, checksum, write_csv, write_json
from _video_review import read_json, read_rows, verify_preview, contact_sheet, selected_windows
from part3_improvement.run_supplementary_ablations import load_sources
from p3_depth.evaluation.temporal import evaluate_relative_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='outputs/shared/video_cache/course_rollercoaster_70')
    parser.add_argument('--run', default='outputs/part1_baseline/dav2_relative/test/course_rollercoaster_70')
    parser.add_argument('--flow', default='outputs/part3_improvement/flow/test/course_rollercoaster_70')
    parser.add_argument('--reference-suite', default='outputs/part3_improvement/test/course_rollercoaster_70')
    parser.add_argument('--output-dir')
    parser.add_argument('--roi', type=int, nargs=4, default=[100, 60, 450, 260])
    a = parser.parse_args()
    cache, run, flow, reference = map(repo_path, (a.cache, a.run, a.flow, a.reference_suite))
    out = repo_path(a.output_dir) if a.output_dir else run/'analysis'
    if out.exists():
        raise FileExistsError('Use a new output directory; preserve prior analysis')
    source = load_sources(cache, run, flow, reference)
    pred, valid = source['base'], source['valid']
    if not np.array_equal(pred, np.load(reference/'E0/depths.npy'), equal_nan=True):
        raise ValueError('Saved E0 does not equal raw DA')
    stats, pairs = evaluate_relative_video(pred, valid, source['fields']['backward'], source['fields']['common_mask'],
                                          source['times'], normalization_limits=source['limits'])
    if stats != read_json(reference/'E0/metrics.json'):
        raise ValueError('E0 diagnostics differ from raw-DA recomputation')
    prior = read_rows(reference/'E0/per_pair.csv')
    if len(prior) != len(pairs):
        raise ValueError('Incomplete saved pair statistics')
    for actual, saved in zip(pairs, prior):
        for k, v in actual.items():
            if v is not None and not np.isclose(v, float(saved[k]), rtol=1e-10, atol=1e-12):
                raise ValueError('Pair statistic differs: '+k)
    preview = verify_preview(cache, run, source['times'], source['limits'])
    frames = np.load(cache/'frames.npy', mmap_mode='r', allow_pickle=False)
    rows = []
    for t, (p, v) in enumerate(zip(pred, valid)):
        q = np.quantile(p[v], [.5, .9, .99])
        rows.append({'frame': t, 'timestamp_s': source['times'][t], 'valid_fraction': float(v.mean()),
                     'zero_fraction': float((v & (p == 0)).mean()), 'median': float(q[0]), 'p90': float(q[1]), 'p99': float(q[2])})
    overview = np.linspace(0, len(pred)-1, 6).round().astype(int).tolist()
    windows = selected_windows(pairs, len(pred))
    out.mkdir(parents=True); figures = out/'figures'; figures.mkdir()
    write_csv(out/'per_pair.csv', pairs); write_csv(out/'per_frame.csv', rows)
    write_json(out/'summary.json', stats)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    axs[0].plot([r['timestamp_s'] for r in pairs], [r['capped_normalized_l1_full_reference'] for r in pairs])
    axs[0].set(title='Flow-aligned variation T (not GT error)', xlabel='Time (s)', ylabel='Normalized change')
    for key in ('median', 'p90', 'p99'):
        axs[1].plot(source['times'], [r[key] for r in rows], label=key)
    axs[1].set(title='Whole-frame raw prediction quantiles', xlabel='Time (s)', ylabel='Relative inverse-like value'); axs[1].legend()
    axs[2].plot(source['times'], [r['zero_fraction'] for r in rows])
    axs[2].set(title='Exactly-zero output fraction', xlabel='Time (s)', ylabel='Fraction of full image')
    for ax in axs: ax.grid(alpha=.2)
    fig.suptitle('DA frame-wise baseline | real motion, scene composition and prediction changes are mixed')
    fig.savefig(figures/'diagnostics.png', dpi=180); plt.close(fig)
    args = (frames, [pred], [valid], [source['limits']], ['DA'], source['times'])
    contact_sheet(*args, overview, figures/'overview.png')
    for window in windows:
        contact_sheet(*args, window, figures/f'consecutive_{window[0]}_{window[-1]}.png')
        contact_sheet(*args, window, figures/f'roi_{window[0]}_{window[-1]}.png', roi=a.roi)
    bm = source['bm']
    resource = {k: bm[k] for k in ('inference_seconds', 'mean_seconds_per_frame', 'fps', 'timing_scope',
                                 'warmup', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')}
    resource['model'] = bm['model']
    write_json(out/'resources.json', resource)
    write_json(out/'quantitative_review.json', {
        'status': 'quantitative_complete_visual_pending', 'frames': len(pred), 'summary': stats,
        'source_metadata_sha256': {p.relative_to(ROOT).as_posix(): checksum(p) for p in
                                  (cache/'metadata.json', run/'metadata.json', flow/'metadata.json', reference/'metadata.json', reference/'verification.json')},
        'preview_checked': preview,
        'analysis_implementation_sha256': {p: checksum(ROOT/p) for p in ('scripts/part1_baseline/analyze_framewise.py', 'scripts/_video_review.py', 'scripts/render_depth_video.py', 'src/p3_depth/evaluation/temporal.py')},
        'overview_frames': overview, 'consecutive_windows': windows, 'roi_xyxy': a.roi,
        'selection': 'Six uniform overview frames; two non-overlapping high-T windows plus fixed middle window. ROI is fixed image coordinates, not object tracking. Post-hoc diagnostic selection.',
        'scope': 'Raw DA output; saved RGB-only flow is used only for post-hoc diagnostics. No fusion, model inference, per-frame display fitting or prediction changes.',
        'limitations': 'No video depth GT. T is not depth accuracy; whole-frame quantile changes also reflect scene composition. Zero relative values are finite model output, not invalid metric depths.',
        'artifacts_sha256': {p.relative_to(out).as_posix(): checksum(p) for p in out.rglob('*') if p.is_file()}})
    print({'frames': len(pred), 'T': stats['capped_normalized_l1_full_reference'], 'windows': windows, 'output': str(out)})


if __name__ == '__main__':
    main()
