"""Prepare one indexed video cache. Register only real project clips before tuning."""

import argparse
import csv
import re
import shutil
import numpy as np

from run_stereo_baseline import ROOT, repo_path, write_csv, write_json, checksum
from p3_depth.data.video import decode_video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--clip-id', required=True)
    p.add_argument('--scene-group', required=True)
    p.add_argument('--split', required=True, choices=['dev','test','smoke'])
    p.add_argument('--scene-type', required=True)
    p.add_argument('--target-fps', type=float, default=10)
    p.add_argument('--max-long-edge', type=int, default=640)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--register', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.clip_id) or not re.fullmatch(r'[A-Za-z0-9_-]+', args.scene_group):
        p.error('Use stable alphanumeric, underscore or hyphen IDs')
    if args.register and (args.split == 'smoke' or args.max_frames is not None):
        p.error('Only complete real dev/test clips may be registered')
    source = repo_path(args.video)
    output = repo_path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new cache directory')
    source_hash = checksum(source)
    fields = ['clip_id','video_path','scene_group','split','scene_type','fps','height','width','duration_s','notes']
    rows = []
    if args.register:
        with (ROOT / 'manifests/videos.csv').open(encoding='utf-8',newline='') as f:
            rows = list(csv.DictReader(f))
        if any(r['clip_id'] == args.clip_id for r in rows):
            raise ValueError('Clip already registered')
        if any(r['scene_group'] == args.scene_group and r['split'] != args.split for r in rows):
            raise ValueError('A scene group cannot cross dev/test')
        if any(source_hash in r['notes'] for r in rows):
            raise ValueError('Identical video content already registered under another ID')
    frames, index, info = decode_video(source, args.target_fps, args.max_long_edge, args.max_frames)
    output.mkdir(parents=True)
    np.save(output / 'frames.npy', frames)
    write_csv(output / 'frames.csv', index)
    meta = {**info, 'clip_id': args.clip_id, 'scene_group': args.scene_group, 'split': args.split,
            'scene_type': args.scene_type, 'source_path': str(source), 'source_sha256': source_hash,
            'frames_sha256': checksum(output / 'frames.npy'), 'index_sha256': checksum(output / 'frames.csv'),
            'frames': len(frames), 'color': 'RGB uint8', 'self_recorded': args.register,
            'scope': 'self_recorded_project' if args.register else 'pipeline_check_not_final_evaluation'}
    write_json(output / 'metadata.json', meta)
    if args.register:
        raw = ROOT / 'data/videos/raw' / (args.clip_id + source.suffix.lower())
        raw.parent.mkdir(parents=True, exist_ok=True)
        if raw.exists() and checksum(raw) != source_hash:
            raise FileExistsError('Different raw video already exists')
        if not raw.exists():
            shutil.copy2(source, raw)
        if checksum(raw) != source_hash:
            raise ValueError('Copied original video hash mismatch')
        rows.append(dict(zip(fields, [args.clip_id, raw.relative_to(ROOT/'data/videos').as_posix(), args.scene_group,
            args.split, args.scene_type, info['source_fps_nominal'], info['source_size_hw'][0], info['source_size_hw'][1],
            info['source_duration_s'], f'source_sha256={source_hash}; cache={output.relative_to(ROOT).as_posix()}'])))
        write_csv(ROOT / 'manifests/videos.csv', rows)
    print(f'PASS {args.clip_id}: {len(frames)} RGB frames, true PTS, shape={frames.shape}; scope={meta["scope"]}')


if __name__ == '__main__':
    main()
