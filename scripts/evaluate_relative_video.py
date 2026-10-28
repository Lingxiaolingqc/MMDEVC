"""Measure Part 1 relative-video fluctuation without converting it to metric depth."""

import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_stereo_baseline import ROOT, repo_path, checksum, write_json, write_csv
from p3_depth.evaluation.temporal import evaluate_relative_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'run', 'flow', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    cache, run, flow, output = [repo_path(x) for x in (args.cache, args.run, args.flow, args.output_dir)]
    if output.exists():
        raise FileExistsError('Use a new diagnostic directory')
    cm, rm, fm = [json.loads((p / 'metadata.json').read_text(encoding='utf-8')) for p in (cache, run, flow)]
    if (rm['status'] != 'complete' or fm['status'] != 'complete' or rm['prediction_type'] != 'inverse_depth_like'
            or rm['unit'] != 'arbitrary' or rm['alignment'] != 'none' or rm['gt_used'] is not False):
        raise ValueError('This diagnostic requires unaligned raw relative predictions, without GT')
    for meta in (rm, fm):
        if any(meta['cache_metadata'][key] != cm[key] for key in ('frames_sha256', 'index_sha256')):
            raise ValueError('Relative prediction and flow must use exactly this RGB/time cache')
    artifacts = [(cache/'frames.npy', cm['frames_sha256']), (cache/'frames.csv', cm['index_sha256']),
                 (run/'depths.npy', rm['depths_sha256']), (run/'valid.npy', rm['valid_sha256'])]
    artifacts += [(flow/(key+'.npy'), fm['artifacts_sha256'][key]) for key in ('backward', 'common_mask')]
    if any(checksum(path) != expected for path, expected in artifacts):
        raise ValueError('Relative diagnostic source file changed')
    arrays = [np.load(path, mmap_mode='r', allow_pickle=False) for path in
              (run/'depths.npy', run/'valid.npy', flow/'backward.npy', flow/'common_mask.npy')]
    with (cache/'frames.csv').open(encoding='utf-8', newline='') as handle:
        times = [float(row['timestamp_s']) for row in csv.DictReader(handle)]
    summary, rows = evaluate_relative_video(*arrays, times, static_scene=cm['scene_type'] == 'static')
    output.mkdir(parents=True)
    write_json(output/'metrics.json', summary)
    write_csv(output/'per_pair.csv', rows)
    write_csv(output/'per_frame.csv', [{'frame': i, 'timestamp_s': t, 'median_relative': median, 'coverage': coverage}
        for i, (t, median, coverage) in enumerate(zip(times, summary['frame_median_relative'], summary['frame_prediction_coverage']))])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(times, summary['frame_median_relative'], color='#217eaa')
    axes[0].set_ylabel('Median inverse-like\n(arbitrary units)')
    axes[1].plot(times[1:], [r['temporal_l1_relative_on_valid'] for r in rows], color='#c46a28')
    axes[1].set_ylabel('Flow-aligned L1\n(arbitrary units)')
    axes[2].plot(times[1:], [r['pair_coverage'] for r in rows], label='Valid / common region')
    pixels = int(np.prod(arrays[0].shape[1:]))
    axes[2].plot(times[1:], [r['reference_pixels']/pixels for r in rows], label='Common region / image')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel('Coverage')
    axes[2].set_xlabel('Original selected timestamp (s)')
    axes[2].legend(loc='lower right')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(f'{cm["clip_id"]} | relative-output diagnostics\n{cm["scope"]} | no GT or per-frame alignment')
    fig.savefig(output/'diagnostics.png', dpi=160)
    plt.close(fig)
    write_json(output/'metadata.json', {'status': 'complete', 'scope': cm['scope'], 'clip_id': cm['clip_id'],
        'scene_type': cm['scene_type'], 'source_metadata_sha256': {
            'cache': checksum(cache/'metadata.json'), 'run': checksum(run/'metadata.json'), 'flow': checksum(flow/'metadata.json')},
        'source_model': rm['model'], 'artifacts_sha256': {name: checksum(output/name) for name in
            ('metrics.json', 'per_pair.csv', 'per_frame.csv', 'diagnostics.png')},
        'implementation_sha256': {name: checksum(ROOT/name) for name in
            ('scripts/evaluate_relative_video.py', 'src/p3_depth/evaluation/temporal.py')}})
    print(f'PASS: {len(times)} relative frames, {summary["reference_pixels"]} fixed reference pixels; scope={cm["scope"]}')


if __name__ == '__main__':
    main()
