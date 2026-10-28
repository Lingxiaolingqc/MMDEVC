"""Package explicit verified video exports, with a separate example-only mode."""

import argparse
import json

from run_stereo_baseline import ROOT, repo_path
from p3_depth.packaging import collect_exports, write_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(repo_path(args.plan).read_text(encoding='utf-8'))
    output = repo_path(args.output)
    if not output.is_relative_to(ROOT/'outputs') or output.suffix != '.zip':
        raise ValueError('Write the local ZIP under outputs/')
    files, manifest = collect_exports(ROOT, plan)
    if args.check_only:
        print(f'PASS preflight: {len(files)} payload files, {len(manifest["clips"])} clips; scope={manifest["scope"]}')
    else:
        print(json.dumps(write_archive(files, manifest, output), indent=2))


if __name__ == '__main__':
    main()
