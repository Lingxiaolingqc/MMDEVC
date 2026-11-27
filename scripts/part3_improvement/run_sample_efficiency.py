"""Run the frozen sample-size/convergence study in isolated serial processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.sample_efficiency.protocol import load,outdir,jobs,name,prepare,check_freeze,freeze
from p3_depth.residual.records import write_json,append_json,now,sha
from p3_depth.residual.locking import study_lock

def memory():
    import ctypes
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]
    m=M();m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {'load_percent':m.load,'physical_available_gib':m.avail_phys/2**30,'commit_available_gib':m.avail_page/2**30}

def record_startup_resources(job):
    """Observational storage estimate only; never changes the frozen training recipe."""
    import shutil
    out=outdir(ROOT);cfg=load(ROOT);completed=[]
    for result_path in (out/'runs').glob('*/result.json'):
        result=json.loads(result_path.read_text())
        if result['status']!='complete':continue
        size=sum(p.stat().st_size for p in result_path.parent.rglob('*') if p.is_file())
        completed.append({'job':name(result['job']),'method':result['job']['method'],'bytes':size})
    same=[x['bytes'] for x in completed if x['method']==job['method']]
    observed=max(same or [x['bytes'] for x in completed] or [0])
    # A conservative planning allowance, not a claim about final file sizes.
    allowance=max(256*2**20,2*observed)
    remaining=len(jobs(cfg))-len(completed)
    run=out/'runs'/name(job);run.mkdir(parents=True,exist_ok=True)
    append_json(run/'startup_resources.jsonl',{
        'at':now(),'job':name(job),'phase':'before_training_process_setup',
        'process_id':os.getpid(),'memory':memory(),'free_disk_bytes':shutil.disk_usage(ROOT).free,
        'estimated_new_job_allowance_bytes':allowance,
        'remaining_incomplete_stages':remaining,'remaining_storage_allowance_bytes':allowance*remaining,
        'estimate_basis':'max(256 MiB, twice largest completed same-method run, or any method if none); raw datasets and shared caches already resident; fixed previews only, no full prediction archive',
        'completed_observed_bytes':completed,'runner_source_sha256':sha(Path(__file__))})

def queue():
    import shutil
    cfg=load(ROOT);out=outdir(ROOT);check_freeze(ROOT)
    path=out/'queue.json'
    entries=[{**j,'status':'pending'} for j in jobs(cfg)]
    if path.exists():
        old=json.loads(path.read_text());assert [name(j) for j in old['jobs']]==[name(j) for j in entries];entries=old['jobs']
    def save():write_json(path,{'updated_at':now(),'jobs':entries,'scope':'development training; test is a later frozen phase'})
    save()
    with study_lock(out):
        for job in entries:
            if job['status']=='complete':continue
            command=[sys.executable,'-X','utf8',str(Path(__file__)),'train','--method',job['method'],'--count',str(job['count']),'--seed',str(job['seed'])]
            for retry in range(2):
                while True:
                    available=memory();disk=shutil.disk_usage(ROOT).free/2**30
                    if available['commit_available_gib']>=cfg['resource']['minimum_available_commit_gib'] and disk>=cfg['resource']['minimum_free_disk_gib']:break
                    job.update(status='waiting_resources',resources=available,free_disk_gib=disk);save();time.sleep(10)
                job.update(status='running',started_at=now());save()
                logs=out/'logs';logs.mkdir(exist_ok=True)
                suffix=str(time.time_ns());stdout=logs/(name(job)+'_'+suffix+'.stdout.log');stderr=logs/(name(job)+'_'+suffix+'.stderr.log')
                with stdout.open('w',encoding='utf8') as a,stderr.open('w',encoding='utf8') as b:
                    worker=subprocess.Popen(command,cwd=ROOT,stdout=a,stderr=b)
                    job['launcher_pid']=worker.pid;job['stdout']=stdout.relative_to(out).as_posix();save()
                    while worker.poll() is None:
                        append_json(out/'resource_samples.jsonl',{'at':now(),'job':name(job),**memory()});time.sleep(10)
                if worker.returncode==0:
                    result=json.loads((out/'runs'/name(job)/'result.json').read_text())
                    assert result['status']=='complete'
                    job.update(status='complete',finished_at=now(),best_abs_rel=result['best']['abs_rel'],stop_step=result['attempted_updates']);save();break
                error=stderr.read_text(encoding='utf8')
                append_json(out/'queue_failures.jsonl',{'at':now(),'job':name(job),'returncode':worker.returncode,'stderr':stderr.relative_to(out).as_posix(),'retry':retry})
                if retry==0 and 'MemoryError' in error and (out/'runs'/name(job)/'resume.pt').exists():
                    job.update(status='recovering_memory_failure');save();time.sleep(20);continue
                job.update(status='failed',finished_at=now());save()
                raise RuntimeError('Worker failed; preserved '+str(stderr))
        print('PASS all 72 development training stages',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','preflight','baseline','train','queue'])
    p.add_argument('--method',choices=['R','L','J','S']);p.add_argument('--count',type=int);p.add_argument('--seed',type=int)
    args=p.parse_args();out=outdir(ROOT);out.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':
        d=prepare(ROOT);print(json.dumps({k:{'images':len(v['ids']),'scenes':v['scene_count']} for k,v in d['subsets'].items()}));return
    if args.phase=='preflight':
        from p3_depth.sample_efficiency.preflight import run
        with study_lock(out):
            report=run(ROOT);freeze(ROOT,report);print('PASS preflight and numerical freeze',flush=True)
    elif args.phase=='baseline':
        from p3_depth.sample_efficiency.training import baseline
        with study_lock(out):print(json.dumps(baseline(ROOT)))
    elif args.phase=='train':
        from p3_depth.sample_efficiency.training import train
        job=dict(method=args.method,count=args.count,seed=args.seed)
        assert job in jobs(load(ROOT)),'Unknown frozen job'
        # Queue parent holds its own lock; child lock prevents direct duplicate workers.
        with study_lock(out/'gpu_worker'):
            record_startup_resources(job)
            print(json.dumps(train(ROOT,job)),flush=True)
    else:queue()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        append_json(outdir(ROOT)/'command_failures.jsonl',{'at':now(),'arguments':sys.argv,'error':repr(exc),'traceback':traceback.format_exc()})
        raise
