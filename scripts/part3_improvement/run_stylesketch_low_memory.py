"""Storage-only recovery runner for the unchanged, frozen SS1 experiment."""
import argparse
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from run_stylesketch_residual import Tee
from p3_depth.stylesketch_residual.common import load, check_freeze
from p3_depth.stylesketch_residual import training, low_memory
from p3_depth.residual.locking import study_lock
from p3_depth.residual.records import sha


def main():
    import json
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['check','train'])
    args = parser.parse_args(); study = load(ROOT); out = ROOT/study['output']
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    check_freeze(ROOT, study)
    with (out/'low_memory.log').open('a', encoding='utf8') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)), study_lock(out):
        if args.phase == 'check':
            low_memory.recovery_check(ROOT, study)
        else:
            record = json.loads((out/'execution_supplement.json').read_text())
            assert all(sha(ROOT/n) == v for n, v in record['runtime_source_sha256'].items())
            # Explicit dependency substitution recorded by execution_supplement.json.
            training.load_pack = low_memory.load_pack
            for seed in study['seeds']:
                training.train(ROOT, study, seed, resume=True)


if __name__ == '__main__': main()
