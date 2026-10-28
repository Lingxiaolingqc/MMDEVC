"""Export RGB, metric depth, point readouts and time traces using original timestamps."""
import argparse
import csv
from fractions import Fraction
import av
import cv2
import numpy as np
from PIL import Image
from run_stereo_baseline import ROOT,repo_path,checksum,write_json
from p3_depth.demo import DemoData

COLORS=[(109,175,255),(99,223,177),(245,191,100),(221,140,245),(251,140,140),(140,229,239)]


def label(image,text,x,y,scale=.55,color=(220,231,245)):
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def make_frame(data,point,index,method):
    canvas=np.full((840,1280,3),(11,18,32),np.uint8)
    label(canvas,'MMDEVC / METRIC DEPTH VIEWER',28,36,.8)
    scope='SELF-RECORDED VIDEO' if data.cache_meta['scope']=='self_recorded_project' else 'EXAMPLE ONLY - not the final self-recorded experiment'
    if any(m['provisional'] for m in data.methods):scope+=' / provisional fusion'
    label(canvas,scope,28,64,.5,(245,191,100))
    label(canvas,f"{data.cache_meta['clip_id']} | t={data.times[index]:.3f} s | frame {index+1}/{data.count}",28,90)
    label(canvas,'RGB',28,119);label(canvas,data.methods[method]['name']+' | fixed display: 0 ... '+str(data.depth_max)+' m',656,119)
    for left,rgb in [(28,data.frames[index]),(656,data.depth_rgb(index,method))]:
        ratio=min(596/data.width,336/data.height)
        w,h=round(data.width*ratio),round(data.height*ratio)
        x0,y0=left+(596-w)//2,128+(336-h)//2
        canvas[y0:y0+h,x0:x0+w]=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_NEAREST)
        cx=x0+round((point['x']+.5)*w/data.width);cy=y0+round((point['y']+.5)*h/data.height)
        cv2.circle(canvas,(cx,cy),6,(0,0,0),3,cv2.LINE_AA)
        cv2.circle(canvas,(cx,cy),6,(255,255,255),1,cv2.LINE_AA)
    label(canvas,f"Fixed pixel ({point['x']}, {point['y']}) - raw camera-axis Z, not Euclidean range",28,493,.6)
    for j,trace in enumerate(point['traces']):
        value=trace['values'][index]
        text='invalid' if value is None else f'{value:.3f} m'
        label(canvas,f"{trace['name']}: {text}",28+j*min(240,1200//len(point['traces'])),526,.6,COLORS[j%len(COLORS)])
    label(canvas,'Fixed-pixel traces (no object tracking); readouts are not clipped to the color scale',28,559,.5)
    left,right,top,bottom=78,1248,582,773
    values=[v for t in point['traces'] for v in t['values'] if v is not None]
    maximum=max(1.,float(np.ceil(max(values,default=0)*1.1)))
    def X(t):return round(left+(t-data.times[0])/max(.001,data.times[-1]-data.times[0])*(right-left))
    def Y(v):return round(bottom-v/maximum*(bottom-top))
    for k in range(5):
        v=maximum*k/4
        cv2.line(canvas,(left,Y(v)),(right,Y(v)),(38,54,76),1)
        label(canvas,f'{v:.1f}',20,Y(v)+5,.45)
    for j,trace in enumerate(point['traces']):
        previous=None
        for t,v in zip(data.times,trace['values']):
            if v is None:previous=None;continue
            xy=(X(t),Y(v))
            if previous is not None:cv2.line(canvas,previous,xy,COLORS[j%len(COLORS)],2,cv2.LINE_AA)
            previous=xy
    cv2.line(canvas,(X(data.times[index]),top),(X(data.times[index]),bottom),(235,240,245),1)
    label(canvas,f'{data.times[0]:.2f} s',left,797,.45)
    label(canvas,f'{data.times[-1]:.2f} s',right-70,797,.45)
    label(canvas,'Gray = invalid. No depth ground truth: this display does not establish absolute accuracy.',28,825,.48)
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--run',action='append',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--method',default='E4')
    p.add_argument('--x',type=int);p.add_argument('--y',type=int)
    p.add_argument('--depth-max',type=float,default=10.)
    args=p.parse_args()
    output=repo_path(args.output_dir)
    if output.exists():raise FileExistsError('Use a new Demo export directory')
    data=DemoData(repo_path(args.cache),[repo_path(x) for x in args.run],args.depth_max)
    try:
        ids=[m['name'] for m in data.methods]
        if args.method not in ids:raise ValueError('Selected display method not provided')
        if len(ids)>6:raise ValueError('At most six methods fit this export layout')
        point=data.point(data.width//2 if args.x is None else args.x,data.height//2 if args.y is None else args.y)
        output.mkdir(parents=True)
        with (output/'point_trace.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['frame_index','timestamp_s','x','y']+ids)
            for i,t in enumerate(data.times):writer.writerow([i,t,point['x'],point['y']]+[trace['values'][i] for trace in point['traces']])
        time_base=Fraction(1,1000000)
        with av.open(str(output/'demo.mp4'),'w') as container:
            stream=container.add_stream('libx264',rate=Fraction(str(data.cache_meta['requested_fps'])))
            stream.width,stream.height,stream.pix_fmt=1280,840,'yuv420p'
            stream.time_base=stream.codec_context.time_base=time_base
            stream.options={'crf':'19','preset':'fast'}
            for i,t in enumerate(data.times):
                rgb=make_frame(data,point,i,ids.index(args.method))
                if i==data.count//2:Image.fromarray(rgb).save(output/'preview.png')
                frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
                frame.pts=round((t-data.times[0])*1000000);frame.time_base=time_base
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        with av.open(str(output/'demo.mp4')) as container:
            decoded=[float(f.pts*f.time_base) for f in container.decode(video=0)]
        if len(decoded)!=data.count or not np.allclose(decoded,np.array(data.times)-data.times[0],rtol=0,atol=2e-6):
            raise ValueError('Demo export changed the selected frame timestamps')
        # Re-read the exported numbers against original arrays, including invalid entries.
        with (output/'point_trace.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        for i,row in enumerate(rows):
            for trace in point['traces']:
                value=trace['values'][i];saved=row[trace['name']]
                if (value is None and saved!='') or (value is not None and float(saved)!=value):
                    raise ValueError('Demo trace does not reproduce raw prediction values')
        write_json(output/'metadata.json',{'status':'complete','scope':data.cache_meta['scope'],
            'clip_id':data.cache_meta['clip_id'],'frames':data.count,'width':1280,'height':840,
            'point':{'x':point['x'],'y':point['y'],'mode':'fixed image coordinate, no object tracking'},
            'display_method':args.method,'depth_max':args.depth_max,'unit':'m','quantity':'camera-axis Z',
            'cache_metadata_sha256':checksum(repo_path(args.cache)/'metadata.json'),
            'methods':[{k:m[k] for k in ('name','metadata_sha256','provisional')} for m in data.methods],
            'verification':{'decoded_timestamps_match':True,'trace_matches_raw_predictions':True},
            'raw_predictions_modified':False,
            'files':{name:checksum(output/name) for name in ['demo.mp4','preview.png','point_trace.csv']},
            'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/render_demo.py','src/p3_depth/demo.py']}})
        print(f'PASS: {data.count} Demo frames; decoded timestamps and raw point trace verified: {output}')
    finally:data.close()


if __name__=='__main__':main()
