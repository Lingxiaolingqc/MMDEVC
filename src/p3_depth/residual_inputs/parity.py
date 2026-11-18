"""Re-score the saved parent checkpoint with extension inputs before new training."""
import json
import torch
from .common import candidate
from .data import prepare_job_features
from .model import make_model
from .evaluation import evaluate
from p3_depth.residual.evaluation import save_evaluation
from p3_depth.residual.records import write_json,sha

def check_reference(root,study,pack,force=False):
    directory=root/study['output']/'reference_validation';path=directory/'parity.json'
    parent=root/study['parent_output']/'runs/frozen_features_seed320112';weight=parent/'best.pt'
    if path.exists() and not force:
        result=json.loads(path.read_text());assert result['checkpoint_sha256']==sha(weight)
        print('PASS reused exact S0 deployment parity',flush=True);return
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    split=json.loads((root/study['split']).read_text());cfg=candidate(study,'S0')
    job={'candidate':cfg,'feature_hw':study['feature_hw'],'fit_ids':split['fit_ids']}
    features,internal,meta=prepare_job_features(root,study,job,pack,pack['original'],{})
    model=make_model(features.shape[1],cfg,internal).cuda()
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model'])
    _,rows=evaluate(model,cfg,pack,features,internal,pack['original'],pack['validation'],8,detailed=True)
    actual=save_evaluation(directory,rows);expected=json.loads((parent/'validation/metrics.json').read_text())
    keys=['abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel']
    differences={key:actual[key]-expected[key] for key in keys}
    assert all(abs(value)<1e-7 for value in differences.values()),differences
    write_json(path,{'status':'pass','checkpoint_sha256':sha(weight),'input_metadata':meta,'differences':differences,'images':218})
    del model;torch.cuda.empty_cache();print('PASS full218 saved S0 metrics '+str(differences),flush=True)
