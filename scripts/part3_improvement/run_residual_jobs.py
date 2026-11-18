"""Run pre-recorded controlled follow-ups sequentially with one shared data pack."""
from pathlib import Path
import argparse
from contextlib import redirect_stdout,redirect_stderr
import gc
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.residual.data import load_pack
from p3_depth.residual.followup import run_job
from p3_depth.residual.jobs import read_job
from p3_depth.residual.locking import study_lock
from p3_depth.residual.size_controls import prepare_size_control


class Tee:
    def __init__(self,terminal,stream):self.terminal,self.stream=terminal,stream
    def write(self,text):self.terminal.write(text);return self.stream.write(text)
    def flush(self):self.terminal.flush();self.stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/residual_study.json')
    parser.add_argument('--jobs',nargs='+',required=True,help='Immutable JSON files saved before viewing their results')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();study=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    paths=[(ROOT/path).resolve() for path in args.jobs]
    records=[read_job(ROOT,study,path) for path in paths]
    if len({x['job']['run_id'] for x in records})!=len(paths):raise ValueError('Duplicate jobs')
    directory=ROOT/study['output']
    with study_lock(directory):
        pack=load_pack(ROOT,study)
        logs=directory/'logs';logs.mkdir(exist_ok=True)
        for path,record in zip(paths,records):
            with (logs/(record['job']['run_id']+'.log')).open('a',encoding='utf-8') as stream:
                with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                    prepare_size_control(ROOT,study,record['job'],pack)
                    run_job(ROOT,study,path,resume=args.resume,shared_pack=pack)
            gc.collect()
            import torch
            torch.cuda.empty_cache()


if __name__=='__main__':main()
