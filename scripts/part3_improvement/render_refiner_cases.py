"""Render fixed and symmetrically ranked test cases from complete retained predictions."""
from pathlib import Path
from contextlib import ExitStack
import argparse
import json
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from analyze_residual_study import read_csv, mean_seed_scores

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.residual.records import sha, write_json
from p3_depth.residual.selection import load_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/residual_study.json')
    args = parser.parse_args()
    study = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    frozen, digest = load_selection(ROOT, study)
    directory = ROOT / study['output'] / 'final_test'
    expected = frozen['baseline_methods'] + [r['run_id'] for r in frozen['refits']]
    results = {name: json.loads((directory / name / 'result.json').read_text(encoding='utf-8')) for name in expected}
    if any(r['status'] != 'complete' or r['spec']['frozen_sha256'] != digest for r in results.values()):
        raise ValueError('All frozen test methods must complete first')
    raw = {name: read_csv(directory / name / 'per_image.csv') for name in expected}
    ids = frozen['test_ids']
    if any({int(r['source_index']) for r in rows} != set(ids) for rows in raw.values()):
        raise ValueError('Test image sets differ')
    score = {name: {int(r['source_index']): r for r in rows} for name, rows in raw.items()}
    for method in ('R0', 'frozen_features'):
        names = [r['run_id'] for r in frozen['refits'] if r['method'] == method]
        if len(names) != 3:
            raise ValueError('Expected all three frozen seeds')
        score[method] = {int(r['source_index']): r for r in mean_seed_scores([raw[n] for n in names])}
    change = {i: float(score['frozen_features'][i]['abs_rel']) - float(score['B0'][i]['abs_rel']) for i in ids}
    ranked = sorted(ids, key=lambda i: (change[i], i))
    reasons = {}
    def select(index, reason):
        reasons.setdefault(index, []).append(reason)
    for index, name in zip([ids[0], ids[len(ids)//2], ids[-1]], ['fixed first', 'fixed middle', 'fixed last']):
        select(index, name)
    for index in ranked[:2]: select(index, 'two largest mean-score improvements')
    for index in ranked[-2:]: select(index, 'two largest mean-score regressions')
    select(ranked[len(ranked)//2], 'median mean-score change')
    # Chosen before new test inference in the final proposal and recorded plan.
    seed = study['screening_seed']
    labels = ['B0', 'B1', f'final_R0_seed{seed}', f'final_frozen_features_seed{seed}']
    output = ROOT / study['output'] / 'analysis/test/cases'
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    with ExitStack() as stack:
        source = stack.enter_context(NYUv2Labeled(ROOT / study['data']))
        files = {name: stack.enter_context(h5py.File(directory / name / 'predictions.h5', 'r')) for name in labels}
        for index, why in reasons.items():
            sample = source.load(index+1)
            predictions = [files[name][str(index)][:] for name in labels]
            if any(x.shape != (480,640) or not np.isfinite(x).all() for x in predictions):
                raise ValueError('Invalid retained prediction')
            _, _, edges, _ = boundary_masks(sample.depth_m, sample.depth_m, sample.valid_gt, sample.valid_gt)
            # A deterministic GT-edge-rich crop for diagnosis, never model selection/input.
            candidates = [(int(edges[y:y+120,x:x+160].sum()), y, x)
                          for y in range(45, 352, 40) for x in range(41, 442, 40)]
            _, y, x = max(candidates, key=lambda v: (v[0], -v[1], -v[2]))
            arrays = [sample.rgb, np.where(sample.valid_gt, sample.depth_m, np.nan), *predictions]
            titles = ['RGB', 'GT', 'Frozen B0', 'Global scale B1', 'R0', 'Internal-feature CNN']
            fig, axes = plt.subplots(2, 6, figsize=(18, 6.8), constrained_layout=True)
            for col, (array, title) in enumerate(zip(arrays, titles)):
                for row in (0, 1):
                    shown = array if row == 0 else array[y:y+120, x:x+160]
                    axes[row,col].imshow(shown, **({} if col == 0 else {'vmin':0, 'vmax':10, 'cmap':'viridis'}))
                    axes[row,col].axis('off')
                if col >= 2:
                    title += '\nAbsRel '+format(float(score[labels[col-2]][index]['abs_rel']), '.4f')
                axes[0,col].set_title(title)
                axes[0,col].add_patch(Rectangle((x,y),160,120,fill=False,edgecolor='yellow',linewidth=1))
            fig.suptitle(f'{sample.sample_id} | '+', '.join(why)+
                         f'\nDepth 0–10 m; displayed seed {seed}; ranking uses mean scores, not an ensemble')
            fig.savefig(output / f'{sample.sample_id}.png', dpi=150)
            plt.close(fig)
            before = np.abs(np.clip(predictions[0], .001, 10)-sample.depth_m)
            after = np.abs(np.clip(predictions[-1], .001, 10)-sample.depth_m)
            fig, axes = plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
            for ax, values, title in zip(axes[:2], [before,after], ['B0 absolute error', 'Refined absolute error']):
                im=ax.imshow(np.where(sample.valid_gt,values,np.nan),vmin=0,vmax=1,cmap='magma')
                ax.set_title(title+' (0–1 m)');ax.axis('off')
            im=axes[2].imshow(np.where(sample.valid_gt,after-before,np.nan),vmin=-.5,vmax=.5,cmap='coolwarm')
            axes[2].set_title('Error change: blue improves, red worsens');axes[2].axis('off')
            fig.colorbar(im,ax=axes[2],label='m',shrink=.7)
            fig.suptitle(sample.sample_id+f' | fixed representative seed {seed}')
            fig.savefig(output/f'{sample.sample_id}_errors.png',dpi=150);plt.close(fig)
            selected.append({'sample_id':sample.sample_id,'source_index':index,'scene':sample.scene,
                             'reasons':why,'crop_yxhw':[y,x,120,160],'mean_score_absrel_change':change[index],
                             'displayed_abs_rel':{name:float(score[name][index]['abs_rel']) for name in labels}})
    fig, axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax, metric in zip(axes, ['abs_rel','boundary_f1']):
        values=[np.array([float(score['frozen_features'][i][metric])-float(score[reference][i][metric]) for i in ids])
                for reference in ['B0','B1','B2','B3','R0']]
        ax.boxplot(values,showfliers=True)
        ax.set_xticks([1,2,3,4,5],['vs B0','vs B1','vs B2','vs B3','vs R0']);ax.axhline(0,color='gray',linestyle='--')
        ax.set_ylabel('Internal-feature CNN minus reference: '+metric)
    fig.suptitle('All654 paired image changes; three-seed mean scores; points beyond1.5IQR shown')
    fig.savefig(output.parent/'paired_distributions.png',dpi=160)
    fig.savefig(output.parent/'paired_distributions.pdf');plt.close(fig)
    write_json(output/'selection.json',{'frozen_sha256':digest,'displayed_seed':seed,'cases':selected,
               'selection_rule':'fixed first/middle/last plus two greatest improvements, two greatest regressions and median mean-score AbsRel change versusB0; overlapping IDs kept once',
               'crop_rule':'highest GT-edge-count120x160 tile on fixed40px grid inside evaluation crop; top-left tie-break',
               'scope':'Post-evaluation diagnosis; no method/parameter/seed selected from test',
               'source_result_sha256':{name:sha(directory/name/'result.json') for name in expected},
               'script_sha256':sha(Path(__file__))})
    print(f'PASS: {len(selected)} symmetrically selected/fixed test cases and full paired distributions: {output}')


if __name__=='__main__':
    main()
