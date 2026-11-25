"""Run S0 then F2/F3/SS1/P2/P3 encoder counterparts in a durable serial queue."""
import argparse
import json
import sys
import traceback
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.encoder_residual.common import load,audit_references,check_freeze
from p3_depth.residual.locking import study_lock
from p3_depth.residual.records import write_json,append_json,now


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','preflight','queue'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--through',choices=['ES0','EF2','EF3','ESS1','EP2','EP3'])
    args=p.parse_args(); study=load(ROOT); out=ROOT/study['output']; out.mkdir(parents=True,exist_ok=True)
    with (out/(args.phase+'.log')).open('a',encoding='utf8') as log,redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)),study_lock(out):
        if args.phase=='prepare':
            if (out/'design_frozen.json').exists(): check_freeze(ROOT,study)
            else:
                from p3_depth.encoder_residual.data import prepare
                audit_references(ROOT,study); prepare(ROOT,study)
        elif args.phase=='preflight':
            from p3_depth.encoder_residual.preflight import run
            run(ROOT,study)
        else:
            from p3_depth.encoder_residual.training import train
            check_freeze(ROOT,study)
            jobs=[{'method':name,'seed':seed,'status':'pending'} for name in study['queue'] for seed in study['seeds']]
            path=out/'queue.json'
            if path.exists():
                saved=json.loads(path.read_text())
                assert [(x['method'],x['seed']) for x in saved['jobs']]==[(x['method'],x['seed']) for x in jobs]
                jobs=saved['jobs']
            def save(): write_json(path,{'updated_at':now(),'order':study['queue'],'jobs':jobs,'scope':'fitting/validation only'})
            save()
            for job in jobs:
                if args.through and study['queue'].index(job['method'])>study['queue'].index(args.through): break
                job.update(status='running',started_at=now()); save()
                try:
                    result=train(ROOT,study,job['method'],job['seed'],args.resume)
                except BaseException as exc:
                    job.update(status='failed',error=repr(exc),at=now()); save(); raise
                job.update(status='complete',best_abs_rel=result['validation']['abs_rel'],finished_at=now()); save()
            print('QUEUE '+json.dumps(jobs),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        out=ROOT/load(ROOT)['output']; out.mkdir(parents=True,exist_ok=True)
        append_json(out/'command_failures.jsonl',{'arguments':sys.argv,'error':repr(exc),'traceback':traceback.format_exc()})
        raise
