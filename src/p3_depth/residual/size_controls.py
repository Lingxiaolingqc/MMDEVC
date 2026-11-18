"""Subset-only scalar controls for the fixed-compute labeled-data experiment."""
import json
from .job_inputs import fit_job_base
from .evaluation import score_image,save_evaluation
from .records import identity,sha,write_json,snapshot


def prepare_size_control(root,study,job,pack):
    if job['phase']!='data_size':return
    output=root/study['output']
    directory=output/'data_size_bases'/f'n{len(job["fit_ids"])}'
    spec={'fit_ids':job['fit_ids'],'validation_ids':job['validation_ids'],
          'cache_sha256':pack['metadata']['cache_sha256'],'baseline':'single scalar times original frozen prediction'}
    path=directory/'result.json'
    if path.exists():
        record=json.loads(path.read_text(encoding='utf-8'))
        if record['spec']!=spec:raise ValueError('Same subset count with changed membership')
        if sha(root/record['per_image_file'])!=record['per_image_sha256']:raise ValueError('Subset baseline evidence changed')
        return record
    directory.mkdir(parents=True,exist_ok=True);snapshot(root,directory)
    # Reuse the already-scored full fitting baseline exactly; smaller subsets
    # must fit their own scalar from only their available labeled images.
    original=json.loads((output/'calibration.json').read_text(encoding='utf-8'))
    if job['fit_ids']==original['fitting_ids'] and original['cache_sha256']==spec['cache_sha256']:
        measured=json.loads((output/'bases/B1/metrics.json').read_text(encoding='utf-8'))
        calibration=original;per_image=output/'bases/B1/per_image.csv';reused=True
    else:
        scalar_job={**job,'candidate':{**job['candidate'],'base':'scale'}}
        base,calibration=fit_job_base(root,study,scalar_job,pack)
        positions={index:i for i,index in enumerate(pack['ids'])}
        measured=save_evaluation(directory,[score_image(base[positions[index]],pack,positions[index]) for index in job['validation_ids']])
        per_image=directory/'per_image.csv';reused=False
    record={'status':'complete','spec':spec,'spec_sha256':identity(spec),'calibration':calibration,'metrics':measured,
            'per_image_file':per_image.relative_to(root).as_posix(),'per_image_sha256':sha(per_image),
            'reused_full_fitting_baseline':reused,'reference_B0':(output/'bases/B0/metrics.json').relative_to(root).as_posix(),
            'scope':'Calibration-only comparison uses the same labeled subset but no neural optimizer updates; not a matched-compute neural training run'}
    write_json(path,record);print(f'PASS subset scalar control n={len(job["fit_ids"])} AbsRel={measured["abs_rel"]:.8f}',flush=True)
    return record
