from pathlib import Path
import json,time,numpy as np
from depth_project.evaluation import prepare_evaluation,edge_retention
from depth_project.experiments import fuse_sequence
from depth_project.flow import estimate_pair
from depth_project.metrics import evaluate_video
root=Path.cwd()
frames,baseline,metadata,_,_,masks,evaluator=prepare_evaluation(root/'third_party/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4',root/'runs/baselines/tokyo-dav2.npz',root/'runs/development')
config=json.loads((root/'configs/final.json').read_text())
rows=[]
original=json.loads((root/'results/development-sweep.json').read_text())['selected']
rows.append({'flow_max_res':640,'temporal':original['temporal'],'edge_retention':original['edge_retention'],'note':'original frozen development result'})
for resolution in [320,480]:
 params={**config,'flow':{**config['flow'],'max_res':resolution}}
 start=time.perf_counter();forward=[];backward=[]
 for a,b in zip(frames,frames[1:]):
  f,bw=estimate_pair(a.rgb,b.rgb,params['flow']);forward.append(f);backward.append(bw)
 flow_s=time.perf_counter()-start
 candidate,_=fuse_sequence(frames,baseline,np.stack(forward),np.stack(backward),params)
 # Evaluation always uses the original frozen 640-resolution backward flow and mask.
 fixed=np.load(root/'runs/development/fixed-flow.npz')
 metric=evaluate_video(candidate,fixed['backward'],masks,evaluator['reference_scale'],metadata['frame_ids'])
 row={'flow_max_res':resolution,'temporal':metric['aligned_mae_normalized'],'edge_retention':edge_retention(candidate,baseline),'flow_compute_s':flow_s,'fixed_evaluation_support':metric['support_pixels']}
 rows.append(row);print(json.dumps(row),flush=True)
threshold=original['temporal']*1.05
eligible=[r for r in rows if r['temporal']<=threshold and r['edge_retention']>=0.90]
chosen=min(eligible,key=lambda r:r['flow_max_res'])
result={'stage':'secondary engineering study after initial held-out evaluation; selection uses development data only','policy':'smallest flow resolution within 5% of original development temporal proxy and at least 90% edge-strength retention','rows':rows,'selected':chosen}
(root/'results/flow-resolution-study.json').write_text(json.dumps(result,indent=2))
efficient={**config,'flow':{**config['flow'],'max_res':chosen['flow_max_res']},'engineering_selection':result['policy'],'stage':result['stage']}
(root/'configs/efficient.json').write_text(json.dumps(efficient,indent=2))
print('ENGINEERING SELECTED',chosen,flush=True)
