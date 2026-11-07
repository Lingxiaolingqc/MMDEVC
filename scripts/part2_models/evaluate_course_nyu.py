"""Re-evaluate a saved raw metric run under the revised course protocol, without inference."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import subprocess
import numpy as np

from _common import ROOT, repo_path, read_yaml, checksum, write_csv, write_json
from p3_depth.evaluation.benchmark import select_nyu
from p3_depth.evaluation.course import PROTOCOL, score_course_nyu, summarize_course


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--paths-config', default='configs/paths.local.yaml')
    args = parser.parse_args()
    run, output = repo_path(args.run), repo_path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new directory; preserve previous scores and raw predictions')
    metadata = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    asset = metadata['model']['asset']
    if (metadata['status'] != 'complete' or metadata['scope'] != 'standard_test_654'
            or asset['prediction_type'] != 'metric_depth' or asset['unit'] != 'm'
            or asset['alignment'] != 'none' or metadata['model']['gt_used_by_model']):
        raise ValueError('Require a completed RGB-only raw metric run on all 654 standard test images')
    manifest, rows = select_nyu(ROOT, 'test')
    if (metadata['identity']['sample_ids'] != [r['sample_id'] for r in rows]
            or metadata['identity']['manifest_sha256'] != checksum(manifest)):
        raise ValueError('Source run differs from the fixed standard split')
    data = repo_path(read_yaml(repo_path(args.paths_config))['data_roots']['nyuv2'])
    records = []
    for row in rows:
        sample = row['sample_id']
        record_path = run / 'records' / (sample + '.json')
        original = json.loads(record_path.read_text(encoding='utf-8'))
        pred_path = run / 'predictions' / (sample + '.npy')
        gt_path, rgb_path = (data / row['depth_gt_path']).resolve(), (data / row['rgb_path']).resolve()
        if not gt_path.is_relative_to(data) or not rgb_path.is_relative_to(data):
            raise ValueError('Manifest input escaped the data directory')
        if original['sample_id'] != sample or original['split'] != 'test':
            raise ValueError('Source record identity changed')
        for path, key in ((pred_path, 'prediction_sha256'), (gt_path, 'gt_sha256'), (rgb_path, 'rgb_sha256')):
            if checksum(path) != original[key]:
                raise ValueError(f'Source data/prediction hash mismatch: {sample} {key}')
        scores = score_course_nyu(np.load(pred_path, allow_pickle=False), np.load(gt_path, allow_pickle=False))
        records.append({'sample_id': sample, **scores,
                        'prediction_sha256': original['prediction_sha256'], 'gt_sha256': original['gt_sha256'],
                        'source_record_sha256': checksum(record_path)})
    summary = summarize_course(records)
    output.mkdir(parents=True)
    write_csv(output / 'per_image.csv', records)
    write_json(output / 'summary.json', summary)
    write_json(output / 'provenance.json', {
        'status': 'complete', 'protocol': PROTOCOL,
        'source_run': run.relative_to(ROOT).as_posix(), 'source_metadata_sha256': checksum(run / 'metadata.json'),
        'source_model': metadata['model'], 'manifest_sha256': checksum(manifest),
        'implementation_sha256': {p: checksum(ROOT / p) for p in
            ['scripts/part2_models/evaluate_course_nyu.py', 'src/p3_depth/evaluation/course.py',
             'src/p3_depth/evaluation/depth.py', 'src/p3_depth/data/nyuv2.py']},
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'per_image_sha256': checksum(output / 'per_image.csv'), 'summary_sha256': checksum(output / 'summary.json'),
        'inference_reused': True, 'raw_predictions_modified': False,
        'efficiency_source': 'Reuse the source inference measurements; this command times no model inference',
    })
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
