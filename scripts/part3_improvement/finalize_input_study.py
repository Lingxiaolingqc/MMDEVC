"""Freeze confirmed input variants, refit all795, then evaluate all654."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p3_depth.residual_inputs.common import load
from p3_depth.residual_inputs.final import freeze,refit,evaluate
from p3_depth.residual.locking import study_lock
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['freeze','refit','evaluate'],required=True);p.add_argument('--names',nargs='+')
    a=p.parse_args();s,_=load(ROOT)
    with study_lock(ROOT/s['output']):
        if a.action=='freeze':
            assert a.names;freeze(ROOT,s,a.names)
        elif a.action=='refit':refit(ROOT,s)
        else:evaluate(ROOT,s)
