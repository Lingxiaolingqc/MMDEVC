"""Serve the saved-depth Demo on loopback; no inference, upload, or dataset writes."""
import argparse
import json
import os
from run_stereo_baseline import ROOT,repo_path,write_json,checksum
from p3_depth.demo import DemoData,create_server


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--run',action='append',required=True)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--depth-max',type=float,default=10.)
    p.add_argument('--session-record',help='Optional local JSON for server URL/PID and source evidence')
    a=p.parse_args()
    data=DemoData(repo_path(a.cache),[repo_path(x) for x in a.run],a.depth_max)
    server=create_server(data,ROOT/'demo',a.port)
    record={'url':f'http://127.0.0.1:{server.server_port}','pid':os.getpid(),
        'cache_metadata_sha256':checksum(repo_path(a.cache)/'metadata.json'),
        'model_metadata_sha256':[m['metadata_sha256'] for m in data.methods],
        'scope':data.cache_meta['scope'],'read_only':True,
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/serve_demo.py','src/p3_depth/demo.py','demo/index.html','demo/app.js','demo/style.css']}}
    if a.session_record:
        path=repo_path(a.session_record)
        if path.exists():
            raise FileExistsError('Use a new session record')
        path.parent.mkdir(parents=True,exist_ok=True);write_json(path,record)
    print(json.dumps(record,ensure_ascii=False),flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        data.close()


if __name__=='__main__':
    main()
