"""Recompute a saved NYUv2 run using GT and saved raw predictions, without model inference."""

import argparse
import json
import numpy as np

from run_stereo_baseline import ROOT, read_yaml, repo_path, checksum
from p3_depth.evaluation.benchmark import score_raw, summarize, select_nyu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--paths-config', default='configs/paths.local.yaml')
    args = p.parse_args()
    run = repo_path(args.run)
    meta = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    if meta['status'] != 'complete':
        raise ValueError('Run is not complete')
    root = repo_path(read_yaml(repo_path(args.paths_config))['data_roots']['nyuv2'])
    records = []
    ids = meta['identity']['sample_ids']
    config = meta['identity']['config']
    manifest, rows = select_nyu(ROOT, config['subset'], config['limit'])
    if checksum(manifest) != meta['identity']['manifest_sha256'] or ids != [r['sample_id'] for r in rows]:
        raise ValueError('Saved IDs differ from the declared fixed split')
    if meta['scope'] == 'standard_test_654' and len(ids) != 654:
        raise ValueError('Incomplete standard test run')
    for sample in ids:
        record = json.loads((run / 'records' / f'{sample}.json').read_text(encoding='utf-8'))
        pp, mp = run / 'predictions' / f'{sample}.npy', run / 'masks' / f'{sample}.npy'
        gp, ip = (root / record['gt_path']).resolve(), (root / record['rgb_path']).resolve()
        if not gp.is_relative_to(root) or not ip.is_relative_to(root):
            raise ValueError('Input path escapes root')
        for path, key in ((pp,'prediction_sha256'), (mp,'mask_sha256'), (gp,'gt_sha256'), (ip,'rgb_sha256')):
            if checksum(path) != record[key]:
                raise ValueError(f'Hash mismatch {path}')
        metrics, transform = score_raw(np.load(pp,allow_pickle=False), np.load(gp,allow_pickle=False), np.load(mp,allow_pickle=False), meta['model']['asset'])
        if metrics != record['metrics'] or transform != record['transformation']:
            raise ValueError(f'Metrics or alignment mismatch {sample}')
        records.append(record)
    if summarize(records) != json.loads((run / 'summary.json').read_text(encoding='utf-8')):
        raise ValueError('Summary differs from saved per-image records')
    print(f'PASS: {len(ids)} saved predictions, source/artifact hashes, alignment, all metrics and summary recomputed')


if __name__ == '__main__':
    main()
