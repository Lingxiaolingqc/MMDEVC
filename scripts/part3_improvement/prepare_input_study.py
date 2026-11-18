"""Cache frozen decoder stages for the bounded Residual input study."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p3_depth.residual_inputs.common import load
from p3_depth.residual_inputs.features import prepare
from p3_depth.residual.locking import study_lock
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['development','test'],default='development');p.add_argument('--limit',type=int)
    a=p.parse_args();s,parent=load(ROOT)
    with study_lock(ROOT/s['output']):prepare(ROOT,s,parent,a.split,a.limit)
