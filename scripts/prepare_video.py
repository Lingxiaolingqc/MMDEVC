"""Prepare a timestamped RGB cache with explicit source identity and scene split."""

import argparse
import csv
import re
import shutil
import numpy as np

from _common import ROOT, repo_path, write_csv, write_json, checksum
from p3_depth.data.video import decode_video
from _paths import choose_output, CURRENT_VIDEO_MANIFEST


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--clip-id', required=True)
    p.add_argument('--scene-group', required=True)
    p.add_argument('--split', required=True, choices=['dev','test','smoke'])
    p.add_argument('--scene-type', required=True)
    p.add_argument('--target-fps', type=float, default=10)
    p.add_argument('--original-fps', action='store_true', help='Keep every consecutive source frame and its PTS')
    p.add_argument('--source-kind', choices=['self_recorded', 'public_example', 'course_example'], default='self_recorded')
    p.add_argument('--source-url', help='Original public source, required for registered public/course examples')
    p.add_argument('--expected-frames', type=int, help='Fail rather than silently accept a short source')
    p.add_argument('--max-long-edge', type=int, default=640)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--output-dir')
    p.add_argument('--manifest', default=CURRENT_VIDEO_MANIFEST)
    registration=p.add_mutually_exclusive_group()
    registration.add_argument('--register', action='store_true', help='Append a new clip to the selected manifest')
    registration.add_argument('--registered', action='store_true', help='Verify an existing manifest entry without changing it')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.clip_id) or not re.fullmatch(r'[A-Za-z0-9_-]+', args.scene_group):
        p.error('Use stable alphanumeric, underscore or hyphen IDs')
    if args.register and args.split == 'smoke':
        p.error('Smoke clips cannot be registered as formal project inputs')
    if args.register and args.source_kind == 'self_recorded' and args.max_frames is not None:
        p.error('Register complete self-recorded clips; use public_example for bounded public excerpts')
    if args.register and args.source_kind != 'self_recorded' and not args.source_url:
        p.error('Registered public inputs need their original source URL')
    source = repo_path(args.video)
    output = choose_output(args.output_dir,'video_cache',clip=args.clip_id)
    if output.exists():
        raise FileExistsError('Use a new cache directory')
    source_hash = checksum(source)
    fields = ['clip_id','video_path','scene_group','split','scene_type','fps','height','width','duration_s','notes']
    rows = []
    manifest=repo_path(args.manifest)
    if args.register or args.registered:
        with manifest.open(encoding='utf-8',newline='') as f:
            rows = list(csv.DictReader(f))
    if args.registered:
        selected=[r for r in rows if r['clip_id']==args.clip_id]
        if len(selected)!=1: raise ValueError('Expected one registered clip')
        row=selected[0];notes=dict(part.strip().split('=',1) for part in row['notes'].split(';'))
        if (any(row[k]!=getattr(args,k) for k in ('split','scene_group','scene_type')) or
                notes['source_sha256']!=source_hash or repo_path(notes['cache'])!=output or
                checksum(ROOT/'data/videos'/row['video_path'])!=source_hash):
            raise ValueError('Source, split or output cache differs from selected registration')
        if (args.split=='smoke' or args.source_kind=='self_recorded' and args.max_frames is not None or
                args.source_kind!='self_recorded' and (not args.source_url or not args.original_fps or args.expected_frames is None)):
            raise ValueError('Registered source requires original FPS, source URL and expected frame count')
    if args.register:
        if any(r['clip_id'] == args.clip_id for r in rows):
            raise ValueError('Clip already registered')
        if any(r['scene_group'] == args.scene_group and r['split'] != args.split for r in rows):
            raise ValueError('A scene group cannot cross dev/test')
        if any(source_hash in r['notes'] for r in rows):
            raise ValueError('Identical video content already registered under another ID')
    frames, index, info = decode_video(source, None if args.original_fps else args.target_fps, args.max_long_edge, args.max_frames)
    if args.expected_frames is not None and len(frames) != args.expected_frames:
        raise ValueError(f'Expected {args.expected_frames} original frames, decoded {len(frames)}; never pad/repeat missing source frames')
    output.mkdir(parents=True)
    np.save(output / 'frames.npy', frames)
    write_csv(output / 'frames.csv', index)
    meta = {**info, 'clip_id': args.clip_id, 'scene_group': args.scene_group, 'split': args.split,
            'scene_type': args.scene_type, 'source_path': str(source), 'source_sha256': source_hash,
            'frames_sha256': checksum(output / 'frames.npy'), 'index_sha256': checksum(output / 'frames.csv'),
            'frames': len(frames), 'color': 'RGB uint8', 'self_recorded': args.source_kind == 'self_recorded',
            'source_kind': args.source_kind, 'source_url': args.source_url, 'expected_frames': args.expected_frames,
            'scope': args.source_kind + '_project' if args.register or args.registered else 'pipeline_check_not_final_evaluation'}
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
        write_csv(manifest, rows)
    print(f'PASS {args.clip_id}: {len(frames)} RGB frames, true PTS, shape={frames.shape}; scope={meta["scope"]}')


if __name__ == '__main__':
    main()
