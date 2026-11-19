"""Prepare or train the predeclared progressive-fusion factorial study."""
import argparse
import gc
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from p3_depth.adaptation_study.common import load, save_job, snapshot
from p3_depth.adaptation_study.data import load_pack
from p3_depth.adaptation_study.residual_training import run_job
from p3_depth.residual.locking import study_lock
from p3_depth.residual_inputs.features import prepare
from p3_depth.residual_inputs.parity import check_reference


class Tee:
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream
    def write(self, value):
        self.terminal.write(value)
        self.stream.write(value)
    def flush(self):
        self.terminal.flush()
        self.stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[320112])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--phase', choices=['factorial','zero_middle','data_size','refit'],default='factorial')
    parser.add_argument('--size',type=int)
    args = parser.parse_args()
    study, parent = load(ROOT)
    with study_lock(ROOT/study['output']):
        if args.prepare:
            prepare(ROOT, study, parent)
            snapshot(ROOT, ROOT/study['output']/'stage_cache/development')
            return
        names = args.names or study['screening']
        assert set(names) <= set(study['candidates'])
        assert set(args.seeds) <= set(study['confirmation_seeds'])
        pack = load_pack(ROOT, study)
        if not args.smoke:
            check_reference(ROOT, study, pack)
        logs = ROOT/study['output']/'logs'
        logs.mkdir(exist_ok=True)
        for seed in args.seeds:
            for name in names:
                phase = 'control' if name in study['controls'] else args.phase
                job = save_job(ROOT, study, name, seed, phase, args.smoke,args.size)
                with (logs/(job.stem+'.log')).open('a', encoding='utf8') as stream:
                    with redirect_stdout(Tee(sys.stdout,stream)), redirect_stderr(Tee(sys.stderr,stream)):
                        run_job(ROOT, study, job, args.resume, pack)
                gc.collect()
                import torch
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
