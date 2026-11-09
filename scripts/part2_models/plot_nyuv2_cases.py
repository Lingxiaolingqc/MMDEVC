"""Render selected NYUv2 RGB/GT/prediction/error panels from saved predictions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from _common import checksum, read_yaml, repo_path, write_json
from p3_depth.data.nyuv2 import evaluation_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs/part2_models/unidepth/test')
    parser.add_argument('--manifest', default='manifests/nyuv2.csv')
    parser.add_argument('--paths-config', default='configs/paths.local.yaml')
    parser.add_argument('--samples', nargs='+', default=['nyuv2_0088', 'nyuv2_0033'])
    parser.add_argument('--depth-max', nargs='+', type=float, default=[10, 2.5])
    parser.add_argument('--error-max', nargs='+', type=float, default=[2, .5])
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    if len(set(args.samples)) != len(args.samples):
        parser.error('Sample IDs must be unique')
    if not len(args.samples) == len(args.depth_max) == len(args.error_max):
        parser.error('Provide one depth/error display maximum per sample')
    if not all(np.isfinite(x) and x > 0 for x in args.depth_max + args.error_max):
        parser.error('Display maxima must be finite and positive')
    run, manifest = repo_path(args.run), repo_path(args.manifest)
    data = repo_path(read_yaml(repo_path(args.paths_config))['data_roots']['nyuv2'])
    output = repo_path(args.output_dir) if args.output_dir else run / 'analysis/selected_figures'
    if output.exists():
        raise FileExistsError('Use a new figure directory; saved results are preserved')
    meta_path = run / 'metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    asset = meta['model']['asset']
    if (meta['status'], asset['prediction_type'], asset['unit'], asset['alignment']) != (
            'complete', 'metric_depth', 'm', 'none'):
        raise ValueError('Require completed, unaligned metric predictions in metres')
    if meta['identity']['manifest_sha256'] != checksum(manifest):
        raise ValueError('Run and current manifest differ')
    with manifest.open(encoding='utf-8', newline='') as handle:
        rows = {r['sample_id']: r for r in csv.DictReader(handle)}
    sources = {str(manifest): checksum(manifest), str(meta_path): checksum(meta_path)}
    samples = []
    for sample in args.samples:
        row = rows[sample]
        if Path(sample).name != sample or '/' in sample or '\\' in sample:
            raise ValueError('Unsafe sample ID')
        rgb_path, gt_path = ((data / row[key]).resolve() for key in ('rgb_path', 'depth_gt_path'))
        if not rgb_path.is_relative_to(data) or not gt_path.is_relative_to(data):
            raise ValueError('Manifest path escapes data root')
        pred_path = run / 'predictions' / (sample+'.npy')
        record_path = run / 'records' / (sample+'.json')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        sources[str(record_path)] = checksum(record_path)
        for path, key in ((rgb_path, 'rgb_sha256'), (gt_path, 'gt_sha256'), (pred_path, 'prediction_sha256')):
            sources[str(path)] = checksum(path)
            if sources[str(path)] != record[key]:
                raise ValueError('Saved sample source changed: '+str(path))
        with Image.open(rgb_path) as image:
            rgb = np.array(image.convert('RGB'))
        gt = np.load(gt_path, allow_pickle=False)
        raw = np.load(pred_path, allow_pickle=False)
        if raw.shape != gt.shape or gt.shape != rgb.shape[:2]:
            raise ValueError('RGB/GT/prediction dimensions differ')
        pred = np.clip(raw, .001, 10)
        mask = evaluation_mask(gt) & np.isfinite(raw) & (raw > 0)
        samples.append((rgb, gt, pred, np.ma.masked_where(~mask, np.abs(pred-gt))))
    output.mkdir(parents=True)
    fig, axes = plt.subplots(len(samples), 4, figsize=(11, 2.25*len(samples)),
                             layout='constrained', squeeze=False)
    for i, ((rgb, gt, pred, error), sample, high, err_high) in enumerate(
            zip(samples, args.samples, args.depth_max, args.error_max)):
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(gt, vmin=0, vmax=high, cmap='viridis')
        axes[i, 2].imshow(pred, vmin=0, vmax=high, cmap='viridis')
        axes[i, 3].imshow(error, vmin=0, vmax=err_high, cmap='magma')
        titles = (sample+' RGB', f'GT (0-{high:g} m)', f'Prediction (0-{high:g} m)', f'|error| (0-{err_high:g} m)')
        for ax, title in zip(axes[i], titles):
            ax.set_title(title, fontsize=10)
            ax.axis('off')
    path = output / 'nyu_cases.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    for source, digest in sources.items():
        if checksum(Path(source)) != digest:
            raise ValueError('Source changed during plotting: '+source)
    write_json(output / 'metadata.json', {
        'samples': args.samples, 'sources_sha256': sources,
        'prediction_clipping_m': [.001, 10], 'alignment': 'none',
        'error_domain': 'course GT crop/range and finite positive prediction; gray/transparent elsewhere',
        'depth_display_max_m': args.depth_max, 'error_display_max_m': args.error_max,
        'figure_sha256': checksum(path), 'implementation_sha256': checksum(Path(__file__)),
        'raw_predictions_modified': False,
    })
    print(f'PASS: {len(samples)} NYUv2 matched cases: {output}')


if __name__ == '__main__':
    main()
