"""Prepare a shared frozen UniDepthV2 cache for the controlled Residual study."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.residual.cache import prepare_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/residual_study.json')
    parser.add_argument('--split', choices=['development', 'test'], default='development')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    study = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    if args.split == 'test':
        from p3_depth.residual.selection import load_selection
        load_selection(ROOT,study)
    if args.limit is not None and args.limit <= 0:
        raise ValueError('Positive smoke limit required')
    from p3_depth.residual.locking import study_lock
    with study_lock(ROOT/study['output']):
        prepare_cache(ROOT, study, args.split, args.limit)


if __name__ == '__main__':
    main()
