"""Run only predeclared input-extension candidates with complete per-run logs."""
import argparse,gc,sys
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p3_depth.residual_inputs.common import load,save_job
from p3_depth.residual_inputs.training import run_job
from p3_depth.residual_inputs.data import load_pack
from p3_depth.residual.locking import study_lock

class Tee:
    def __init__(self,terminal,stream):self.terminal,self.stream=terminal,stream
    def write(self,value):self.terminal.write(value);self.stream.write(value)
    def flush(self):self.terminal.flush();self.stream.flush()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--names',nargs='+');p.add_argument('--seeds',nargs='+',type=int,default=[320112])
    p.add_argument('--phase',choices=['screen','confirm','ablation','conditional','refit'],default='screen')
    p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--recheck-reference',action='store_true',help='Recompute all218 saved S0 scores before this queue')
    a=p.parse_args();s,parent=load(ROOT);names=a.names or s['screening']
    assert set(names)<=set(s['candidates']) and set(a.seeds)<=set(s['confirmation_seeds'])
    with study_lock(ROOT/s['output']):
        pack=load_pack(ROOT,s);logs=ROOT/s['output']/'logs';logs.mkdir(exist_ok=True)
        if not a.smoke:
            from p3_depth.residual_inputs.parity import check_reference
            check_reference(ROOT,s,pack,force=a.recheck_reference)
        for seed in a.seeds:
            for name in names:
                job=save_job(ROOT,s,name,seed,a.phase,a.epochs,a.smoke)
                with (logs/(job.stem+'.log')).open('a',encoding='utf8') as stream:
                    with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                        run_job(ROOT,s,job,a.resume,pack)
                gc.collect()
                import torch
                torch.cuda.empty_cache()
if __name__=='__main__':main()
