"""Study identity, nested scene coverage, early stopping and deterministic batches."""
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from p3_depth.residual.records import sha, write_json, now

CONFIG = 'configs/sample_efficiency_20261127.json'

def load(root):
    return json.loads((root/CONFIG).read_text(encoding='utf8'))

def outdir(root):
    return root/load(root)['output']

def jobs(cfg):
    values=[]
    for count in cfg['primary_order']:
        for method in cfg['methods']:
            for seed in cfg['seeds']: values.append(dict(method=method,count=count,seed=seed))
    for count in cfg['stage_first']:
        for seed in cfg['seeds']: values.append(dict(method='S',count=count,seed=seed))
    for method in cfg['methods']:
        for seed in cfg['seeds']: values.append(dict(method=method,count=577,seed=seed))
    for count in cfg['stage_last']:
        for seed in cfg['seeds']: values.append(dict(method='S',count=count,seed=seed))
    assert len(values)==72
    return values

def name(job):
    return f"{job['method']}_n{job['count']}_seed{job['seed']}"

def make_subsets(split, meta, seed):
    scene_by_id=dict(zip(meta['ids'],meta['scenes']))
    groups=defaultdict(list)
    for i in split['fit_ids']: groups[scene_by_id[i]].append(i)
    assert len(groups)==187 and len(split['fit_ids'])==577
    assert not set(groups)&{scene_by_id[i] for i in split['validation_ids']}
    rng=random.Random(seed)
    by_type=defaultdict(list)
    for scene in sorted(groups): by_type[scene.rsplit('_',1)[0]].append(scene)
    assert len(by_type)==22
    for scenes in by_type.values(): rng.shuffle(scenes)
    types=sorted(by_type); rng.shuffle(types)
    # Guarantee one of every type, then greedily approach its scene proportion.
    chosen=[by_type[t].pop(0) for t in types]
    totals={t:len(by_type[t])+1 for t in types}; allocated={t:1 for t in types}
    while len(chosen)<100:
        candidates=[t for t in types if by_type[t]]
        t=max(candidates,key=lambda t:(100*totals[t]/187-allocated[t],-types.index(t)))
        chosen.append(by_type[t].pop(0));allocated[t]+=1
    remaining=[s for t in types for s in by_type[t]];rng.shuffle(remaining)
    scene_order=chosen+remaining
    for scene in scene_order:
        groups[scene].sort();rng.shuffle(groups[scene])
    order=[]
    for round_index in range(max(map(len,groups.values()))):
        order.extend(groups[s][round_index] for s in scene_order if len(groups[s])>round_index)
    assert len(order)==len(set(order))==577
    subsets={}
    for count in [100,200,300,400,500,577]:
        ids=order[:count];scenes={scene_by_id[i] for i in ids}
        assert len(scenes)==(100 if count==100 else 187)
        subsets[str(count)]={'ids':ids,'scenes':sorted(scenes),'scene_count':len(scenes),
          'scene_types':sorted({s.rsplit('_',1)[0] for s in scenes})}
    return {'ordering_seed':seed,'ordered_fit_ids':order,'subsets':subsets,
            'validation_ids':split['validation_ids'],'test_ids':split['test_ids']}

def batch_order(ids,seed,steps=6000,batch_size=8):
    import numpy as np
    pending=[];epoch=0
    for _ in range(steps):
        while len(pending)<batch_size:
            pending.extend(np.random.default_rng(seed+epoch).permutation(ids).tolist());epoch+=1
        yield pending[:batch_size]
        pending=pending[batch_size:]

def update_stop(state,score,step,training):
    state=dict(state)
    if not math.isfinite(score): raise ValueError('Nonfinite validation score')
    if score<state.get('best',float('inf')):
        state.update(best=score,best_step=step)
    anchor=state.get('anchor',float('inf'))
    significant=score<anchor and (not math.isfinite(anchor) or (anchor-score)/anchor>=training['significant_relative_improvement'])
    if significant: state.update(anchor=score,stale=0)
    elif step>0: state['stale']=state.get('stale',0)+1
    state['stop']=step>=training['minimum_steps'] and state.get('stale',0)>=training['patience_evaluations']
    return state

def convergence(history,cfg):
    vals=[h for h in history if 'validation' in h]
    best=min(h['validation']['abs_rel'] for h in vals)
    end=vals[-1]['step'];stable=None
    for i,h in enumerate(vals):
        if end-h['step']>=cfg['stable_span_steps'] and all(v['validation']['abs_rel']<=best*(1+cfg['stable_relative_tolerance']) for v in vals[i:]):
            stable=h['step'];break
    targets={}
    n=cfg['target_consecutive_evaluations']
    for threshold in cfg['targets']:
        targets[str(threshold)]=next((vals[i]['step'] for i in range(len(vals)-n+1)
            if all(v['validation']['abs_rel']<=threshold for v in vals[i:i+n])),None)
    return {'best_step':min(vals,key=lambda h:h['validation']['abs_rel'])['step'],
            'stop_step':end,'stable_step':stable,'stable_at_end':stable is not None,'targets':targets}

def prepare(root):
    cfg=load(root);out=root/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/cfg['split']).read_text())
    disk=root/cfg['disk_inputs'];meta=json.loads((disk/'metadata.json').read_text())
    stage=root/cfg['features'];fm=json.loads((stage/'metadata.json').read_text())
    assert meta['ids']==fm['ids']==sorted(split['fit_ids']+split['validation_ids'])
    assert meta['split_sha256']==fm['split_sha256']==sha(root/cfg['split'])
    design=make_subsets(split,meta,cfg['subset_seed'])
    design.update(split_sha256=sha(root/cfg['split']),dataset_sha256=split['dataset_sha256'])
    path=out/'subsets.json'
    if path.exists(): assert json.loads(path.read_text())==design
    else: write_json(path,design)
    paths={str(root/cfg['data']):split['dataset_sha256']}
    paths.update({str(disk/n):h for n,h in meta['sha256'].items()})
    paths.update({str(stage/(n+'.npy')):fm['sha256'][n+'.npy'] for n in ['init','mid','late']})
    audit=out/'data_audit.json'
    if audit.exists():
        old=json.loads(audit.read_text())
        for p,record in old['files'].items():
            stat=Path(p).stat();assert stat.st_size==record['size'] and stat.st_mtime_ns==record['mtime_ns']
        assert old['expected']==paths
    else:
        records={}
        for p,h in paths.items():
            actual=sha(Path(p));assert actual==h,p
            stat=Path(p).stat();records[p]={'sha256':h,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        write_json(audit,{'at':now(),'expected':paths,'files':records})
    return design

def source_files(root):
    paths=[root/CONFIG,root/load(root)['plan']]
    paths+=list((root/'src/p3_depth').rglob('*.py'))
    return sorted(paths)

def freeze(root,preflight):
    import zipfile
    cfg=load(root);out=root/cfg['output'];target=out/'frozen.json'
    record={'at':now(),'config':cfg,'subsets_sha256':sha(out/'subsets.json'),
            'data_audit_sha256':sha(out/'data_audit.json'),'preflight':preflight,
            'sources':{p.relative_to(root).as_posix():sha(p) for p in source_files(root)}}
    if target.exists(): return check_freeze(root)
    with zipfile.ZipFile(out/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in record['sources']:z.write(root/p,p)
    write_json(target,record);return record

def check_freeze(root):
    out=outdir(root);r=json.loads((out/'frozen.json').read_text())
    assert r['config']==load(root)
    for p,h in r['sources'].items():assert sha(root/p)==h,('Frozen source changed',p)
    for p,k in [('subsets.json','subsets_sha256'),('data_audit.json','data_audit_sha256')]:
        assert sha(out/p)==r[k]
    audit=json.loads((out/'data_audit.json').read_text())
    for p,v in audit['files'].items():
        stat=Path(p).stat();assert (stat.st_size,stat.st_mtime_ns)==(v['size'],v['mtime_ns']),p
    return r
