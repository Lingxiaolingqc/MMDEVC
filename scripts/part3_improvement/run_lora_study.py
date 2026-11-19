"""Run bounded LoRA jobs after the Residual design has been frozen."""
import argparse
import gc
import sys
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.adaptation_study.common import load
from p3_depth.adaptation_study.lora_training import candidates,prepare_pack,save_job,run
from p3_depth.residual.locking import study_lock
from run_fusion_ablations import Tee


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--names',nargs='+');p.add_argument('--seeds',type=int,nargs='+',default=[320112])
    p.add_argument('--phase',choices=['search','confirm','data_size','refit'],default='search')
    p.add_argument('--size',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();study,_=load(ROOT)
    names=a.names or list(candidates(study))
    assert set(names)<=set(candidates(study)) and set(a.seeds)<=set(study['confirmation_seeds'])
    with study_lock(ROOT/study['output']):
        jobs=[save_job(ROOT,study,name,seed,a.phase,a.size,a.smoke) for seed in a.seeds for name in names]
        pack=prepare_pack(ROOT,study);logs=ROOT/study['output']/'logs';logs.mkdir(exist_ok=True)
        for job in jobs:
            with (logs/(job.stem+'.log')).open('a',encoding='utf8') as stream:
                with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                    run(ROOT,study,job,pack,a.resume)
            gc.collect()
            import torch
            torch.cuda.empty_cache()


if __name__=='__main__':main()
