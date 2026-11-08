"""Recheck saved stereo/LR results; export error distributions and visual evidence.

No matcher inference or parameter changes. Boxes use exact valid-pixel quantiles,
5th/95th percentile whiskers and separate coverage; all tails remain in CSVs.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from _common import ROOT, repo_path, checksum, read_yaml, select_sample, write_csv, write_json
from p3_depth.data.middlebury import EXPECTED_SCENES, load_sample, disparity_to_depth_m
from p3_depth.models.stereo_sgbm import decode_disparity, left_right_consistency
from p3_depth.evaluation.stereo import evaluate_region, compare_lr_rejection

BINS = (0, 2, 3, 4, 6, np.inf)
BIN_LABELS = ('0-2', '2-3', '3-4', '4-6', '6+')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(path, keys):
    with path.open(encoding='utf-8', newline='') as f:
        return {tuple(r[k] for k in keys): r for r in csv.DictReader(f)}


def agree(actual, expected, label):
    if isinstance(actual, (bool, np.bool_)):
        ok = str(actual).lower() == str(expected).lower()
    elif actual is None:
        ok = expected in ('', None)
    else:
        ok = np.isclose(actual, float(expected), rtol=1e-9, atol=1e-10)
    if not ok:
        raise ValueError(f'{label}: recomputed {actual}, saved {expected}')


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    names = ('p05', 'q25', 'median', 'q75', 'p95', 'p99', 'maximum')
    if not values.size:
        return {k: None for k in ('mean', 'rmse', *names)}
    return {'mean': float(values.mean()), 'rmse': float(np.sqrt(np.mean(values**2))),
            **dict(zip(names, map(float, np.quantile(values, [.05, .25, .5, .75, .95, .99, 1]))))}


def box(row, prefix=''):
    return {name: row[prefix+source] for name, source in
            [('whislo', 'p05'), ('q1', 'q25'), ('med', 'median'), ('q3', 'q75'), ('whishi', 'p95'), ('mean', 'mean')]}


def colors(values, lo, hi, cmap='viridis'):
    good = np.isfinite(values)
    q = np.clip((np.where(good, values, lo)-lo)/(hi-lo), 0, 1)
    rgb = np.rint(plt.get_cmap(cmap)(q)[..., :3]*255).astype(np.uint8)
    rgb[~good] = 185
    return rgb


def overview(scenes, target):
    width, height = 260, 200
    canvas = Image.new('RGB', (width*6, len(scenes)*height), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('arial.ttf', 13)
    except OSError:
        font = ImageFont.load_default(size=13)
    for i, (sid, rgb, gt, before, after, removed, upper) in enumerate(scenes):
        error = np.abs(after.astype(float)-gt)
        error[~np.isfinite(gt)] = np.nan
        rejected = np.zeros((*removed.shape, 3), dtype=np.uint8)
        old_error = np.abs(before.astype(float)-gt)
        rejected[removed & ~np.isfinite(gt)] = 185
        rejected[removed & np.isfinite(gt) & (old_error > 1)] = (40, 210, 90)
        rejected[removed & np.isfinite(gt) & (old_error <= 1)] = (245, 70, 70)
        images = [rgb, colors(gt, 0, upper), colors(before, 0, upper), colors(after, 0, upper),
                  colors(error, 0, 3, 'magma'), rejected]
        labels = ['RGB', f'GT disparity 0-{upper}px', 'LR off (same scale)', 'LR on (same scale)',
                  '|disp error| 0-3px (cap)', 'Rejected: green >1px, red <=1px']
        for j, (img, label) in enumerate(zip(images, labels)):
            pic = Image.fromarray(img)
            pic.thumbnail((width, height-36), Image.Resampling.LANCZOS)
            canvas.paste(pic, (j*width+(width-pic.width)//2, i*height+36))
            draw.text((j*width+3, i*height+2), sid, fill='black', font=font)
            draw.text((j*width+3, i*height+18), label, fill='black', font=font)
    canvas.save(target)


def render_distributions(distributions, bins, figures):
    fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True, layout='constrained')
    for state, delta, color in [('off', -.15, '#b78a43'), ('on', .15, '#287c9f')]:
        rows = [r for r in distributions if r['custom_lr'] == state]
        positions = np.arange(len(rows))+delta
        for ax, prefix in zip(axs[:2], ('disparity_', 'depth_')):
            ax.bxp([box(r, prefix) for r in rows], positions=positions, widths=.24, vert=False,
                   showfliers=False, showmeans=True, manage_ticks=False,
                   boxprops={'color': color}, medianprops={'color': color}, whiskerprops={'color': color},
                   capprops={'color': color}, meanprops={'marker': '+', 'markeredgecolor': color})
        axs[2].plot([r['coverage']*100 for r in rows], positions, 'o', color=color, label='LR '+state)
    axs[0].set_yticks(np.arange(15), [s+(' (dev)' if s in ('Adirondack', 'Pipes') else '') for s in EXPECTED_SCENES])
    axs[0].invert_yaxis()
    axs[0].set_xscale('symlog', linthresh=.1); axs[1].set_xscale('symlog', linthresh=.01)
    for ax, label in zip(axs, ('Absolute disparity error (px)', 'Absolute Z-depth error (m)', 'GT coverage (%)')):
        ax.set_xlabel(label); ax.grid(alpha=.2)
    axs[0].set_xlim(left=0); axs[1].set_xlim(left=0)
    axs[2].legend()
    fig.suptitle('15 scenes, all GT | conditional valid-pixel errors + coverage\nBoxes Q1-Q3; median; whiskers P5-P95; + mean; tails retained in CSV; symlog error axes')
    fig.savefig(figures/'scene_errors_and_coverage.png', dpi=180); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for i, group in enumerate(('all_15_descriptive', 'evaluation')):
        for state, delta, color in [('off', -.15, '#b78a43'), ('on', .15, '#287c9f')]:
            rows = [r for r in bins if r['split'] == group and r['custom_lr'] == state]
            positions = np.arange(5)+delta
            populated = [(j, r) for j, r in enumerate(rows) if r['valid_pixels']]
            axs[i, 0].bxp([box(r) for j, r in populated], positions=[positions[j] for j, r in populated], widths=.24,
                          showfliers=False, showmeans=True, manage_ticks=False, boxprops={'color': color},
                          medianprops={'color': color}, whiskerprops={'color': color}, capprops={'color': color},
                          meanprops={'marker': '+', 'markeredgecolor': color})
            axs[i, 1].plot(positions, [r['coverage']*100 if r['coverage'] is not None else np.nan for r in rows],
                           'o-', color=color, label='LR '+state)
        axs[i, 0].set_yscale('symlog', linthresh=.01)
        axs[i, 0].set_ylim(bottom=0)
        axs[i, 0].set(title=group, ylabel='Absolute Z-depth error (m)')
        axs[i, 1].set(title=group, ylabel='GT coverage (%)'); axs[i, 1].legend()
        for ax in axs[i]:
            ax.set_xticks(np.arange(5), BIN_LABELS); ax.set_xlabel('GT Z-distance bin (m)'); ax.grid(alpha=.2)
    fig.suptitle('Distance bins pool pixels; P5-P95 whiskers, Q1-Q3 boxes, + mean\n6+ m contains only Pipes (dev); empty bins have no estimate. Distance and scene content are confounded.')
    fig.savefig(figures/'depth_bins.png', dpi=180); plt.close(fig)


def depth_geometry(sample, prediction, target):
    """Pipes needs a wider display scale than the old common 0-4 m preview."""
    gt = np.where(sample.valid_gt, sample.depth_m, np.nan)
    error = np.abs(prediction.astype(float)-gt)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout='constrained')
    axes[0].imshow(sample.left_rgb); axes[0].set_title('Pipes RGB')
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#b9b9b9')
    for ax, values, title, upper in zip(axes[1:], (gt, prediction, error),
                                      ('GT-derived Z', 'SGBM Z after LR', 'Absolute Z error (cap at 1m)'), (9, 9, 1)):
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=upper)
        ax.set_title(title); fig.colorbar(im, ax=ax, label='m', shrink=.7)
    for ax in axes: ax.axis('off')
    fig.suptitle('Pipes | GT/prediction display 0-9 m | gray invalid | no array clipping or alignment')
    fig.savefig(target, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='outputs/part1_baseline/stereo/all')
    p.add_argument('--lr-run', default='outputs/part1_baseline/stereo/lr_analysis')
    p.add_argument('--manifest', default='manifests/middlebury.csv')
    p.add_argument('--paths-config', default='configs/paths.local.yaml')
    p.add_argument('--output-dir')
    a = p.parse_args()
    run, lr, manifest = map(repo_path, (a.run, a.lr_run, a.manifest))
    out = repo_path(a.output_dir) if a.output_dir else run/'analysis'
    if out.exists():
        raise FileExistsError('Preserve earlier analysis; use a new --output-dir')
    data = repo_path(read_yaml(repo_path(a.paths_config))['data_roots']['middlebury'])
    old_main = table(run/'per_scene.csv', ('sample_id', 'region'))
    old_lr = table(lr/'per_scene.csv', ('sample_id', 'region', 'custom_lr'))
    old_reject = table(lr/'rejections.csv', ('sample_id', 'region'))
    lm = read(lr/'metadata.json')
    if lm['status'] != 'complete' or lm['manifest_sha256'] != checksum(manifest):
        raise ValueError('Incomplete or stale LR source')
    # LR analysis added a rejection helper after the original baseline; its
    # evaluator version is the relevant provenance for this saved comparison.
    for name, digest in lm['implementation_sha256'].items():
        if checksum(ROOT/name) != digest:
            raise ValueError('LR analysis implementation changed: '+name)
    for name, digest in lm['files_sha256'].items():
        if checksum(lr/name) != digest:
            raise ValueError('LR evidence changed: '+name)
    lr_sources = {r['sample_id']: r['metadata_sha256'] for r in lm['sources']}
    metrics, distributions, rejections, conversions, source_hashes, views, bin_values = [], [], [], [], {}, [], {}
    for sid in EXPECTED_SCENES:
        folder = run/sid
        m = read(folder/'metadata.json')
        if m['status'] != 'completed' or m['manifest_sha256'] != checksum(manifest) or checksum(folder/'metadata.json') != lr_sources[sid]:
            raise ValueError('Incomplete or changed source: '+sid)
        for item in m['sources']:
            if checksum(data/item['path']) != item['sha256']:
                raise ValueError('Input changed: '+sid)
        for name, digest in m['artifacts_sha256'].items():
            if checksum(folder/name) != digest:
                raise ValueError('Saved stereo artifact changed: '+sid+'/'+name)
        for name in ('src/p3_depth/data/middlebury.py', 'src/p3_depth/models/stereo_sgbm.py'):
            if checksum(ROOT/name) != m['implementation_sha256'][name]:
                raise ValueError('Stereo implementation changed: '+name)
        paths, split = select_sample(manifest, data, sid)
        s = load_sample(paths)
        cfg = read_yaml(folder/'config_resolved.yaml')
        arrays = lambda name: np.load(folder/name, allow_pickle=False)
        before = arrays('diagnostics/left_before_lr_px.npy')
        right = arrays('diagnostics/right_signed_px.npy')
        after, valid, depth = (arrays(f'{sub}/{sid}.npy') for sub in ('disparities', 'masks', 'predictions'))
        for raw, decoded, minimum in [('left_fixed16.npy', before, cfg['min_disparity']),
                                       ('right_fixed16.npy', right, cfg['right_min_disparity'])]:
            if not np.array_equal(decode_disparity(arrays('diagnostics/'+raw), minimum), decoded, equal_nan=True):
                raise ValueError('Fixed16 decode differs: '+sid)
        passed, residual = left_right_consistency(before, right, cfg['lr_max_diff_px'])
        bz = disparity_to_depth_m(before, s.calibration)
        bv = np.isfinite(before) & np.isfinite(bz) & (bz > 0)
        expected = passed & bv
        if not (np.array_equal(expected, valid) and np.array_equal(residual, arrays('diagnostics/lr_error_px.npy'), equal_nan=True)
                and np.array_equal(after, np.where(valid, before, np.nan), equal_nan=True)
                and np.array_equal(depth, disparity_to_depth_m(after, s.calibration), equal_nan=True)):
            raise ValueError('Saved LR/depth does not recompute: '+sid)
        gt = np.isfinite(s.disparity_px) & (s.disparity_px > 0)
        if not np.array_equal(gt, s.valid_gt) or not np.array_equal(gt, arrays('diagnostics/evaluation_all_gt.npy')):
            raise ValueError('Course GT mask differs: '+sid)
        if not np.array_equal(gt & s.nonoccluded, arrays('diagnostics/evaluation_nonoccluded_gt.npy')):
            raise ValueError('Nonoccluded mask differs: '+sid)
        removed = bv & ~valid
        if not np.array_equal(removed, np.load(lr/'rejected_masks'/f'{sid}.npy')):
            raise ValueError('Rejected mask differs: '+sid)
        for region_name, region in [('all_gt', gt), ('nonoccluded_gt', gt & s.nonoccluded)]:
            rej = compare_lr_rejection(before, after, bv, valid, s.disparity_px, region)
            rej.update(removed_no_comparable_right=int((removed & region & ~np.isfinite(residual)).sum()),
                       removed_residual_above_threshold=int((removed & region & np.isfinite(residual)).sum()))
            for key, value in rej.items():
                agree(value, old_reject[(sid, region_name)][key], f'{sid} {key}')
            rejections.append({'sample_id': sid, 'split': split, 'region': region_name, **rej})
            for state, dp, zp, vp in [('off', before, bz, bv), ('on', after, depth, valid)]:
                score = evaluate_region(zp, dp, vp, s.depth_m, s.disparity_px, region)
                flat = {k: v for k, v in score.items() if k != 'depth_on_valid_predictions'}
                flat.update({k+'_on_valid': v for k, v in score['depth_on_valid_predictions'].items()})
                row = {'sample_id': sid, 'split': split, 'region': region_name, 'custom_lr': state, **flat}
                for key in old_lr[(sid, region_name, state)]:
                    if key in flat:
                        agree(flat[key], old_lr[(sid, region_name, state)][key], f'{sid} LR {key}')
                if state == 'on':
                    for key in old_main[(sid, region_name)]:
                        if key in flat:
                            agree(flat[key], old_main[(sid, region_name)][key], f'{sid} main {key}')
                metrics.append(row)
                if region_name != 'all_gt':
                    continue
                use = region & vp
                de = np.abs(dp[use].astype(float)-s.disparity_px[use])
                ze = np.abs(zp[use].astype(float)-s.depth_m[use])
                distributions.append({'sample_id': sid, 'split': split, 'custom_lr': state,
                                      'gt_pixels': int(region.sum()), 'valid_pixels': int(use.sum()), 'coverage': float(use.sum()/region.sum()),
                                      **{'disparity_'+k: v for k, v in distribution(de).items()},
                                      **{'depth_'+k: v for k, v in distribution(ze).items()}})
                for group in ('all_15_descriptive', 'evaluation'):
                    if group == 'evaluation' and split != 'evaluation':
                        continue
                    for j, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
                        region_bin = region & (s.depth_m >= lo) & (s.depth_m < hi)
                        keep = region_bin & vp
                        entry = bin_values.setdefault((group, state, j), {'gt': 0, 'errors': [], 'scenes': []})
                        entry['gt'] += int(region_bin.sum())
                        entry['errors'].append(np.abs(zp[keep].astype(float)-s.depth_m[keep]))
                        if region_bin.any():
                            entry['scenes'].append(sid)
        use = gt & valid
        d, g = after[use].astype(float), s.disparity_px[use].astype(float)
        c, offset = s.calibration.focal_px*s.calibration.baseline_mm/1000, s.calibration.doffs_px
        exact_z_error = np.abs(c/(d+offset)-c/(g+offset))
        propagated = c*np.abs(d-g)/np.abs((d+offset)*(g+offset))
        if not np.allclose(exact_z_error, propagated, rtol=1e-10, atol=1e-12):
            raise ValueError('Depth error identity failed: '+sid)
        conversions.append({'sample_id': sid, 'split': split, 'focal_px': s.calibration.focal_px,
                            'baseline_mm': s.calibration.baseline_mm, 'doffs_px': offset,
                            'gt_depth_min_m': float(s.depth_m[gt].min()), 'gt_depth_max_m': float(s.depth_m[gt].max()),
                            'max_float64_error_identity_residual_m': float(np.abs(exact_z_error-propagated).max()),
                            'max_float32_saved_depth_error_residual_m': float(np.abs(np.abs(depth[use].astype(float)-s.depth_m[use])-propagated).max()),
                            'prediction_seconds': m['timing']['prediction_seconds'], 'timing_benchmark': m['timing']['benchmark']})
        source_hashes[(folder/'metadata.json').relative_to(ROOT).as_posix()] = checksum(folder/'metadata.json')
        views.append((sid, s.left_rgb, np.where(gt, s.disparity_px, np.nan), np.where(bv, before, np.nan), after, removed, cfg['num_disparities']))
        if sid == 'Pipes':
            pipes_sample, pipes_prediction = s, depth
    summaries = []
    for group in ('all_15_descriptive', 'dev', 'evaluation'):
        for region in ('all_gt', 'nonoccluded_gt'):
            for state in ('off', 'on'):
                selected = [r for r in metrics if r['region'] == region and r['custom_lr'] == state and (group == 'all_15_descriptive' or r['split'] == group)]
                n, k = sum(r['gt_pixels'] for r in selected), sum(r['valid_prediction_pixels'] for r in selected)
                row = {'split': group, 'region': region, 'custom_lr': state, 'scenes': len(selected), 'gt_pixels': n,
                       'valid_pixels': k, 'pooled_coverage': k/n,
                       'macro_disparity_mae_px': float(np.mean([r['disparity_mae_px_on_valid'] for r in selected])),
                       'macro_depth_rmse_m': float(np.mean([r['rmse_m_on_valid'] for r in selected])),
                       'pooled_bad1_invalid_as_bad': sum(r['bad1_fraction_invalid_as_bad']*r['gt_pixels'] for r in selected)/n}
                summaries.append(row)
    bins = []
    for (group, state, j), item in bin_values.items():
        err = np.concatenate(item['errors'])
        bins.append({'split': group, 'custom_lr': state, 'gt_distance_bin_m': BIN_LABELS[j], 'gt_pixels': item['gt'],
                     'valid_pixels': len(err), 'coverage': len(err)/item['gt'] if item['gt'] else None,
                     'contributing_scenes': '|'.join(item['scenes']), **distribution(err)})
    out.mkdir(parents=True); figures = out/'figures'; figures.mkdir()
    for name, rows in [('per_scene', metrics), ('error_distributions', distributions), ('rejections', rejections),
                       ('conversion_checks', conversions), ('summary', summaries), ('depth_bins', bins)]:
        write_csv(out/(name+'.csv'), rows)
    render_distributions(distributions, bins, figures)
    depth_geometry(pipes_sample, pipes_prediction, figures/'Pipes_depth_geometry.png')
    for start in range(0, 15, 5):
        overview(views[start:start+5], figures/f'scenes_{start+1:02d}_{start+5:02d}.png')
    source_hashes.update({p.relative_to(ROOT).as_posix(): checksum(p) for p in
                         (manifest, run/'per_scene.csv', lr/'metadata.json', lr/'summary.csv')})
    write_json(out/'quantitative_review.json', {'status': 'quantitative_complete_visual_pending', 'scenes': 15,
        'verification': 'All source and saved artifact hashes; fixed16 decode; LR masks/residual; saved depth conversion; course/nonoccluded GT; 60 LR rows and 30 main rows; rejection statistics; float64 depth-error identity.',
        'source_sha256': source_hashes, 'analysis_script_sha256': checksum(Path(__file__)),
        'summary': summaries, 'box_definition': 'Exact valid-pixel P5/Q1/median/Q3/P95; + mean; symlog axes; tails recorded in CSV. Not confidence intervals.',
        'bin_definition': 'Left-closed, right-open GT Z-distance bins [0,2,3,4,6,infinity), pooled pixels; contributing scene IDs in CSV.',
        'conversion': 'Z=f_px*(baseline_mm/1000)/(d_px+doffs_px); GT Z derives from GT disparity, not independent sensor GT.',
        'comparison': 'Only extra custom LR differs. OpenCV internal disp12MaxDiff=1 stays fixed. Retained disparities unchanged.',
        'visual_scope_pending': 'All 15 overview rows; detailed existing native scene/LR diagnostic figures to inspect separately.',
        'source_predictions_modified': False, 'new_inference': False,
        'artifacts_sha256': {p.relative_to(out).as_posix(): checksum(p) for p in out.rglob('*') if p.is_file()}})
    print(json.dumps({'scenes': 15, 'rows': len(metrics), 'summary': summaries[:4], 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
