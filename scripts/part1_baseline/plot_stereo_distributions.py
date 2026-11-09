"""Plot selected stereo error distributions from the saved full-pixel statistics."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from _common import checksum, repo_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', default='outputs/part1_baseline/stereo/all/analysis')
    parser.add_argument('--samples', nargs='+', default=['Adirondack', 'Jadeplant', 'PianoL', 'Pipes', 'Teddy'])
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    source = repo_path(args.analysis) / 'error_distributions.csv'
    output = repo_path(args.output_dir) if args.output_dir else source.parent / 'selected_figures'
    if output.exists():
        raise FileExistsError('Use a new figure directory; saved results are preserved')
    with source.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    if len(set(args.samples)) != len(args.samples):
        raise ValueError('Sample IDs must be unique')
    for sample in args.samples:
        matches = [r for r in rows if r['sample_id'] == sample and r['custom_lr'] == 'on']
        if len(matches) != 1:
            raise ValueError('Expected one LR-on distribution: ' + sample)
        row = matches[0]
        if not 0 <= float(row['coverage']) <= 1:
            raise ValueError('Invalid GT coverage')
        for prefix in ('disparity', 'depth'):
            quantiles = [float(row[prefix + '_' + key]) for key in ('p05', 'q25', 'median', 'q75', 'p95')]
            if not np.isfinite(quantiles).all() or np.any(np.diff(quantiles) < 0):
                raise ValueError('Expected finite ordered quantiles: ' + sample)
        selected.append(row)
    source_hash = checksum(source)
    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.9), layout='constrained')
    for ax, prefix, unit in zip(axes, ('disparity', 'depth'), ('px', 'm')):
        items, labels = [], []
        for row in selected:
            items.append(dict(med=float(row[prefix+'_median']), q1=float(row[prefix+'_q25']),
                              q3=float(row[prefix+'_q75']), whislo=float(row[prefix+'_p05']),
                              whishi=float(row[prefix+'_p95']), fliers=[]))
            labels.append(row['sample_id'] + '\n' + f"{100*float(row['coverage']):.1f}%")
        ax.bxp(items, showfliers=False)
        ax.set_xticks(range(1, len(selected)+1), labels, fontsize=11)
        ax.set_ylabel('Absolute error ('+unit+')', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=.2)
    path = output / 'stereo_boxes.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    if checksum(source) != source_hash:
        raise ValueError('Distribution source changed during plotting')
    write_json(output / 'metadata.json', {
        'source': str(source), 'source_sha256': source_hash, 'samples': args.samples,
        'custom_lr': 'on', 'error_domain': 'valid predictions within all-GT domain',
        'boxes': 'Q1-Q3 with median and P5-P95 whiskers; tails remain in source CSV',
        'coverage': 'valid prediction pixels / fixed GT pixels',
        'figure_sha256': checksum(path), 'implementation_sha256': checksum(Path(__file__)),
        'raw_predictions_modified': False,
    })
    print(f'PASS: {len(selected)} stereo distributions: {output}')


if __name__ == '__main__':
    main()
