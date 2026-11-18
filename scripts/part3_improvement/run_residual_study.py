"""Evaluate frozen bases or train one controlled Residual candidate with retained records."""
from pathlib import Path
import argparse
import json
import sys
from contextlib import redirect_stdout, redirect_stderr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.residual.training import evaluate_bases, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/residual_study.json')
    parser.add_argument('--action', choices=['bases', 'edges', 'train', 'screen', 'confirm'], required=True)
    parser.add_argument('--candidate', default='R0')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--candidates', nargs='+', help='Bounded subset of the recorded screening matrix')
    parser.add_argument('--selection',default='configs/residual_confirmation.json',help='Recorded shortlist used by --action confirm')
    args = parser.parse_args()
    study = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    from p3_depth.residual.locking import study_lock
    with study_lock(ROOT/study['output']):
        run_action(study,args)


def run_action(study,args):
    if args.action == 'bases':
        evaluate_bases(ROOT, study)
    elif args.action == 'edges':
        from p3_depth.residual.edges import prepare_edges
        prepare_edges(ROOT, study)
    elif args.action in ('screen','confirm'):
        from p3_depth.residual.data import load_pack
        import gc
        import torch
        if args.action=='confirm':
            from p3_depth.residual.records import sha
            from p3_depth.residual.model import candidate
            decision=json.loads((ROOT/args.selection).read_text(encoding='utf-8'))
            names=decision['methods'];seeds=decision['seeds']
            if args.candidates or args.seed or args.smoke:
                raise ValueError('Confirmation settings come only from the recorded selection')
            if decision['training']!=study['training'] or decision['feature_hw']!=study['feature_hw'] or decision['split_sha256']!=sha(ROOT/study['split']):
                raise ValueError('Confirmation protocol changed after selection')
            if decision['candidate_configs']!={name:candidate(name) for name in names}:
                raise ValueError('Confirmation candidate definitions changed')
        else:
            names = args.candidates or study['screening']
            seeds=[args.seed or study['screening_seed']]
        if not set(names).issubset(study['screening']) or len(set(names)) != len(names):
            raise ValueError('Screening candidates must be unique entries in the recorded matrix')
        if args.smoke:
            raise ValueError('Run smoke checks individually')
        pack = load_pack(ROOT, study)
        logs = ROOT / study['output'] / 'logs'
        logs.mkdir(exist_ok=True)
        class Tee:
            def __init__(self, terminal, stream):
                self.terminal, self.stream = terminal, stream
            def write(self, text):
                self.terminal.write(text)
                return self.stream.write(text)
            def flush(self):
                self.terminal.flush()
                self.stream.flush()
        for seed in seeds:
            for name in names:
                with (logs / f'{name}_seed{seed}.log').open('a', encoding='utf-8') as stream:
                    with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                        run_training(ROOT, study, name, seed, resume=args.resume, shared_pack=pack)
                gc.collect()
                torch.cuda.empty_cache()
    else:
        run_training(ROOT, study, args.candidate, args.seed or study['screening_seed'], args.smoke, args.resume)


if __name__ == '__main__':
    main()
