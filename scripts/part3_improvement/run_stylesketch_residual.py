"""Run the single predeclared StyleSketch-inspired Residual on fitting/validation only."""
import argparse
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from p3_depth.stylesketch_residual.common import load, audit_references
from p3_depth.residual.locking import study_lock
from p3_depth.residual.records import append_json


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'preflight', 'train'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    study = load(ROOT)
    out = ROOT/study['output']
    out.mkdir(parents=True, exist_ok=True)
    with (out/(args.phase+'.log')).open('a', encoding='utf8') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)), study_lock(out):
        if args.phase == 'prepare':
            from p3_depth.stylesketch_residual.data import prepare
            audit_references(ROOT, study)
            prepare(ROOT, study)
        elif args.phase == 'preflight':
            from p3_depth.stylesketch_residual.preflight import run
            run(ROOT, study)
        else:
            from p3_depth.stylesketch_residual.training import train
            selected = args.seeds or study['seeds']
            assert set(selected) <= set(study['seeds']) and len(set(selected)) == len(selected)
            for seed in selected:
                train(ROOT, study, seed, args.resume)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        append_json(ROOT/load(ROOT)['output']/'command_failures.jsonl',
                    {'arguments': sys.argv, 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
