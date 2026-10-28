"""Read-only local depth inspection using verified saved metric predictions."""
import csv
from functools import lru_cache
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
from pathlib import Path
from urllib.parse import parse_qs,urlparse
import cv2
import numpy as np
from PIL import Image

from p3_depth.models.runtime import file_hash

LABELS={'E0':'E0 · 原始逐帧深度','E1':'E1 · EMA','E2':'E2 · 光流融合',
        'E3':'E3 · 光流与遮挡过滤','E4':'E4 · 自适应光流融合',
        'dav2_metric':'DA V2 · 原始逐帧深度','vda_metric':'VDA · 视频模型'}


class DemoData:
    def __init__(self,cache,runs,depth_max=10.):
        self.cache=Path(cache)
        if not np.isfinite(depth_max) or depth_max<=0:
            raise ValueError('Positive finite display maximum required')
        self.depth_max=float(depth_max)
        self.cache_meta=json.loads((self.cache/'metadata.json').read_text(encoding='utf-8'))
        for name,key in [('frames.npy','frames_sha256'),('frames.csv','index_sha256')]:
            if file_hash(self.cache/name)!=self.cache_meta[key]:
                raise ValueError('RGB/time cache hash mismatch')
        self.frames=np.load(self.cache/'frames.npy',mmap_mode='r',allow_pickle=False)
        with (self.cache/'frames.csv').open(encoding='utf-8',newline='') as f:
            self.times=[float(r['timestamp_s']) for r in csv.DictReader(f)]
        if self.frames.dtype!=np.uint8 or self.frames.ndim!=4 or self.frames.shape[-1]!=3:
            raise ValueError('Expected THWC uint8 RGB cache')
        self.count,self.height,self.width=self.frames.shape[:3]
        if len(self.times)!=self.count or not np.isfinite(self.times).all() or not np.all(np.diff(self.times)>0):
            raise ValueError('Invalid video timestamps')
        self.methods=[]
        for i,directory in enumerate(runs):
            path=Path(directory)
            meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
            if meta['status']!='complete' or meta['prediction_type']!='metric_depth' or meta['unit']!='m' or meta['alignment']!='none':
                raise ValueError('The metric Demo requires completed, unaligned Z depth in metres')
            if any(meta['cache_metadata'][k]!=self.cache_meta[k] for k in ('frames_sha256','index_sha256')):
                raise ValueError('Each method must use exactly the displayed RGB/time cache')
            for name,key in [('depths.npy','depths_sha256'),('valid.npy','valid_sha256')]:
                if file_hash(path/name)!=meta[key]:
                    raise ValueError('Depth artifact hash mismatch')
            depth=np.load(path/'depths.npy',mmap_mode='r',allow_pickle=False)
            valid=np.load(path/'valid.npy',mmap_mode='r',allow_pickle=False)
            if depth.shape!=self.frames.shape[:3] or not np.issubdtype(depth.dtype,np.floating) or valid.shape!=depth.shape or valid.dtype!=np.bool_:
                raise ValueError('Depth/mask dimensions must match RGB frames')
            name=meta['model']['model']
            self.methods.append({'id':i,'name':name,'label':LABELS.get(name,name),
                'provisional':meta.get('config_status')=='provisional','depth':depth,'valid':valid,
                'metadata_sha256':file_hash(path/'metadata.json')})
        if not self.methods or len({m['name'] for m in self.methods})!=len(self.methods):
            raise ValueError('Provide at least one method, with unique method names')

    def public_metadata(self):
        return {'clip_id':self.cache_meta['clip_id'],'scope':self.cache_meta['scope'],
            'frames':self.count,'width':self.width,'height':self.height,'timestamps':self.times,
            'depth_max':self.depth_max,'unit':'m','quantity':'camera-axis Z depth',
            'methods':[{k:m[k] for k in ('id','name','label','provisional')} for m in self.methods]}

    def close(self):
        """Release mapped files after the HTTP server has stopped using this instance."""
        self.image_bytes.cache_clear()
        arrays=[self.frames]+[m[key] for m in self.methods for key in ('depth','valid')]
        for array in arrays:
            if getattr(array,'_mmap',None) is not None:
                array._mmap.close()

    def point(self,x,y):
        if not (0<=x<self.width and 0<=y<self.height):
            raise ValueError('Point outside original cached image')
        traces=[]
        for method in self.methods:
            values=method['depth'][:,y,x]
            valid=method['valid'][:,y,x] & np.isfinite(values) & (values>0)
            traces.append({'id':method['id'],'name':method['name'],
                'values':[float(v) if ok else None for v,ok in zip(values,valid)]})
        return {'x':x,'y':y,'timestamps':self.times,'traces':traces,
                'interpretation':'fixed image coordinate; not object tracking; raw Z values, not display-clipped'}

    def depth_rgb(self,index,method):
        if not 0<=index<self.count or not 0<=method<len(self.methods):
            raise ValueError('Invalid frame or method')
        m=self.methods[method];d=m['depth'][index]
        valid=m['valid'][index] & np.isfinite(d) & (d>0)
        scaled=np.rint(np.clip(np.where(valid,d,0)/self.depth_max,0,1)*255).astype(np.uint8)
        rgb=cv2.cvtColor(cv2.applyColorMap(scaled,cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
        rgb[~valid]=180
        return rgb

    @lru_cache(maxsize=96)
    def image_bytes(self,index,method):
        if not 0<=index<self.count:
            raise ValueError('Invalid frame')
        rgb=self.frames[index] if method==-1 else self.depth_rgb(index,method)
        buf=io.BytesIO();Image.fromarray(rgb).save(buf,format='PNG')
        return buf.getvalue()


def create_server(data,assets,port=0):
    assets=Path(assets)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send(self,status,body,content_type):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            request=urlparse(self.path);q=parse_qs(request.query)
            try:
                if request.path in ('/','/app.js','/style.css'):
                    file,kind={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}[request.path]
                    return self.send(200,(assets/file).read_bytes(),kind)
                if request.path=='/api/meta':
                    payload=data.public_metadata()
                elif request.path=='/api/point':
                    payload=data.point(int(q['x'][0]),int(q['y'][0]))
                elif request.path=='/api/frame':
                    return self.send(200,data.image_bytes(int(q['index'][0]),int(q['method'][0])),'image/png')
                else:
                    return self.send(404,b'Not found','text/plain')
                self.send(200,json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8'),'application/json; charset=utf-8')
            except (KeyError,ValueError,IndexError):
                self.send(400,b'Invalid frame, method or coordinate','text/plain')
            except (BrokenPipeError,ConnectionResetError):
                pass

    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
