"""Build report panels directly from saved course arrays; record sources and crops."""
import argparse
import csv
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from run_stereo_baseline import ROOT,repo_path,checksum,write_json
from render_depth_video import colorize
from p3_depth.application import background_blur


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);a=p.parse_args()
    output=repo_path(a.output_dir)
    if output.exists():raise FileExistsError('Use a new figure directory')
    output.mkdir(parents=True);sources={};font=ImageFont.truetype('arial.ttf',25)
    def load(path):
        sources[str(path.relative_to(ROOT)).replace('\\','/')]=checksum(path)
        return np.load(path,mmap_mode='r',allow_pickle=False)
    def panel(arrays,labels,width=400):
        height=round(arrays[0].shape[0]*width/arrays[0].shape[1])
        canvas=Image.new('RGB',(width*len(arrays),height+32),'white');draw=ImageDraw.Draw(canvas)
        for i,(array,label) in enumerate(zip(arrays,labels)):
            canvas.paste(Image.fromarray(array).resize((width,height)),(i*width,32))
            draw.text((i*width+5,5),label,fill='black',font=font)
        return canvas
    def stack(rows,name):
        out=Image.new('RGB',(rows[0].width,sum(x.height for x in rows)),'white');y=0
        for row in rows:out.paste(row,(0,y));y+=row.height
        out.save(output/name)
    root=ROOT/'outputs/course_video_20261102';runs={}
    for clip,suite in [('dev_tokyo_100','alpha070'),('course_rollercoaster_70','frozen_suite')]:
        base=root/clip;rgb=load(base/'cache/frames.npy')
        depths={v:load(base/suite/v/'depths.npy') for v in ('E0','E1','E4')}
        depths['VDA']=load(base/'vda_relative_364/depths.npy')
        meta=json.loads((base/suite/'metadata.json').read_text(encoding='utf-8'))
        limits=meta['baseline_normalization_limits'];vlimits=list(map(float,np.quantile(depths['VDA'],[.01,.99])))
        runs[clip]=(rgb,depths,limits,vlimits)
    rgb,depths,limits,vlimits=runs['course_rollercoaster_70']
    rows=[]
    for t in (35,69):
        arrays=[rgb[t]]+[colorize(depths[v][t],np.isfinite(depths[v][t]),*(vlimits if v=='VDA' else limits)) for v in ('E0','E4','VDA')]
        rows.append(panel(arrays,[f'RGB | {t/24:.3f} s','DA V2 relative','E4 (same DA limits)','VDA (own limits)']))
    stack(rows,'video_comparison.png')
    cases=[('dev_tokyo_100',45,(200,160,390,350),'Dev: moving person'),
           ('course_rollercoaster_70',35,(130,60,300,170),'Test: riders / rails'),
           ('course_rollercoaster_70',69,(430,215,640,360),'Test: distant detail')]
    rows=[]
    for clip,t,(x0,y0,x1,y1),label in cases:
        rgb,depths,limits,vlimits=runs[clip]
        arrays=[rgb[t]]+[colorize(depths[v][t],np.isfinite(depths[v][t]),*(vlimits if v=='VDA' else limits)) for v in ('E0','E1','E4','VDA')]
        rows.append(panel([x[y0:y1,x0:x1] for x in arrays],[label,'DA V2','E1 EMA','E4 fusion','VDA'],width=320))
    # Different crop aspect ratios are retained, with a common column width.
    stack(rows,'difficult_cases.png')
    with (ROOT/'manifests/nyuv2.csv').open(encoding='utf-8',newline='') as f:nyu={x['sample_id']:x for x in csv.DictReader(f)}
    with (ROOT/'results/course_20261102/nyuv2/per_image.csv').open(encoding='utf-8',newline='') as f:scores=sorted(csv.DictReader(f),key=lambda x:(float(x['rmse_m']),x['sample_id']))
    selected=[scores[(len(scores)-1)//2],scores[-1]];rows=[]
    for record in selected:
        name=record['sample_id'];row=nyu[name];rgbpath=ROOT/'data/nyuv2'/row['rgb_path']
        sources[rgbpath.relative_to(ROOT).as_posix()]=checksum(rgbpath)
        rgb=np.array(Image.open(rgbpath).convert('RGB'));gt=load(ROOT/'data/nyuv2'/row['depth_gt_path'])
        pred=np.clip(load(ROOT/'outputs/nyuv2_test_20261028/unidepth/predictions'/f'{name}.npy'),.001,10)
        rows.append(panel([rgb,colorize(gt,np.isfinite(gt)&(gt>0),0,10),colorize(pred,np.isfinite(pred),0,10)],
                          [name,'Filled GT (0-10 m)',f'UniDepth | RMSE {float(record["rmse_m"]):.3f} m']))
    stack(rows,'nyu_course.png')
    app=ROOT/'outputs/course_video_20261102/course_rollercoaster_70/application/metadata.json'
    sources[app.relative_to(ROOT).as_posix()]=checksum(app)
    rgb,depths,limits,_=runs['course_rollercoaster_70'];t=35
    raw_blur,_=background_blur(rgb[t],depths['E0'][t],np.isfinite(depths['E0'][t]),limits)
    fused_blur,_=background_blur(rgb[t],depths['E4'][t],np.isfinite(depths['E4'][t]),limits)
    panel([rgb[t],colorize(depths['E4'][t],np.isfinite(depths['E4'][t]),*limits),raw_blur,fused_blur],
          ['RGB | 1.458 s','E4 relative depth','Background blur: DA','Background blur: E4']).save(output/'application.png')
    write_json(output/'provenance.json',{'sources_sha256':sources,'nyu_selection':'lower median and maximum revised-course per-image RMSE',
        'nyu_selected_ids':[r['sample_id'] for r in selected],'video_frames':[35,69],
        'crops':[{'clip_id':c,'frame':t,'xyxy':box,'label':label} for c,t,box,label in cases],
        'outputs_sha256':{f.name:checksum(f) for f in output.glob('*.png')},'implementation_sha256':checksum(ROOT/'scripts/build_course_figures.py')})
    print('PASS: report figures generated from saved predictions; source and crop provenance recorded')


if __name__=='__main__':main()
