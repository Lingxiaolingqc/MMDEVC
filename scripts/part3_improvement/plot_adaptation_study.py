"""Plot independent Residual and LoRA validation studies without comparing the winners."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.adaptation_study.common import load
from p3_depth.residual.records import sha, write_json


def save(fig, destination, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(destination / (stem + '.' + extension), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['factorial', 'lora'], required=True)
    args = parser.parse_args()
    study, _ = load(ROOT)
    out = ROOT / study['output']
    path = out / 'analysis' / args.family / 'review.json'
    review = json.loads(path.read_text())
    rows = review['summary']
    destination = path.parent / 'figures'
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, key, label in zip(axes, ['abs_rel', 'boundary_band_abs_rel', 'boundary_f1'],
                             ['Validation AbsRel (lower better)', 'Boundary-band AbsRel (lower better)',
                              'Boundary F1 (higher better)']):
        x = np.arange(len(rows))
        ax.bar(x, [r[key] for r in rows], color='#4c789f', alpha=.8)
        for i, row in enumerate(rows):
            if row[key + '_sd'] is not None:
                ax.errorbar(i, row[key], yerr=row[key + '_sd'], color='black', capsize=3)
        ax.set_xticks(x, [r['method'] + f" (n={r['n_seeds']})" for r in rows], rotation=60, ha='right')
        ax.set_title(label)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'{args.family}: independent validation study; error bars = training-seed SD, not confidence intervals')
    save(fig, destination, 'summary')
    pairs = review['pairs']
    if args.family == 'factorial':
        pairs = [p for p in pairs if p['factor'] != 'reference']
    if pairs:
        fig, ax = plt.subplots(figsize=(9, max(4, len(pairs) * .45)), constrained_layout=True)
        for i, pair in enumerate(pairs):
            values = pair['metrics']['abs_rel']
            lo, hi = values['scene_cluster_bootstrap_ci95']
            value = values['candidate_minus_reference']
            ax.plot([lo, hi], [i, i], color='#1f6a8a', lw=2)
            ax.plot(value, i, 'o', color='#1f6a8a')
        seed_counts = {row['method']: row['n_seeds'] for row in rows}
        labels = ([p['after'] + ' - ' + p['before'] + f" (n={len(p['seeds'])})" for p in pairs]
                  if args.family == 'factorial' else
                  [p['method'] + ' - UniDepth' + f" (n={seed_counts[p['method']]})" for p in pairs])
        ax.set_yticks(range(len(pairs)), labels)
        ax.axvline(0, color='black', lw=1, linestyle='--')
        ax.set(xlabel='Paired change in validation AbsRel (negative is better)',
               title='Scene-cluster bootstrap 95% intervals; n = training seeds\nExploratory; no multiplicity adjustment')
        ax.invert_yaxis()
        save(fig, destination, 'paired_changes')
    histories = []
    for result_path in sorted((out / 'followups').glob('*/result.json')):
        result = json.loads(result_path.read_text())
        if result['smoke']:
            continue
        if args.family == 'factorial' and result['phase'] not in ('factorial', 'control', 'zero_middle'):
            continue
        if args.family == 'lora' and result.get('family') != 'lora':
            continue
        histories.append((result_path.parent.name, json.loads((result_path.parent / 'history.json').read_text())))
    for run, history in histories:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        validation = [r for r in history if 'validation' in r]
        axes[0].plot([r['updates_attempted'] for r in history], [r['train_loss'] for r in history], label='Training loss')
        axes[0].plot([r['updates_attempted'] for r in validation], [r['validation']['abs_rel'] for r in validation], label='Validation AbsRel')
        axes[0].set(xlabel='Attempted updates', ylabel='AbsRel / objective')
        axes[0].legend()
        axes[1].plot([r['image_exposures'] for r in validation], [r['validation']['abs_rel'] for r in validation])
        axes[1].set(xlabel='Training image exposures', ylabel='Validation AbsRel')
        fig.suptitle(run)
        save(fig, destination, run + '_curve')
    write_json(destination / 'index.json', {
        'review_sha256': sha(path), 'family': args.family,
        'scope': 'Validation only; independent model selection; no Residual-versus-LoRA comparison.',
        'files': {p.name: sha(p) for p in sorted(destination.iterdir()) if p.suffix in ('.png', '.pdf')}
    })
    print('PASS independent study figures: ' + str(destination))


if __name__ == '__main__':
    main()
