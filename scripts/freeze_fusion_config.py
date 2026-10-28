"""Freeze reviewed development candidates; never choose the lowest error automatically."""

import argparse
import re
import yaml

from run_stereo_baseline import ROOT, repo_path, write_json, write_csv, checksum
from p3_depth.temporal.selection import selection_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True, metavar='NAME=SUITE',
                        help='Repeat once per development clip and candidate')
    parser.add_argument('--selected', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = repo_path(args.output_dir)
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError('Use a new output directory inside the repository')
    candidates = {}
    for item in args.candidate:
        name, suite = item.split('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Candidate names use letters, digits, underscores or hyphens')
        candidates.setdefault(name, []).append(suite)
    proof, rows = selection_evidence(ROOT, candidates, args.selected, repo_path(args.review))
    output.mkdir(parents=True)
    write_json(output / 'selection.json', proof)
    write_csv(output / 'development_comparison.csv', rows)
    config = {'schema_version': 1, 'status': 'frozen',
              'selection_evidence': {'path': (output / 'selection.json').relative_to(ROOT).as_posix(),
                                     'sha256': checksum(output / 'selection.json')},
              'parameters': proof['selected_parameters']}
    (output / 'fusion_frozen.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'PASS: froze {args.selected} using {len(proof["dev_clip_ids"])} dev clips; reviewer={proof["reviewer_kind"]}')


if __name__ == '__main__':
    main()
