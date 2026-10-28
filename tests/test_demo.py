"""Demo must show original metric values, gray invalid pixels and matching timestamps."""
from pathlib import Path
import io
import json
import sys
import tempfile
import threading
import unittest
from urllib.request import build_opener,ProxyHandler
from urllib.error import HTTPError
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.demo import DemoData,create_server
from p3_depth.models.runtime import file_hash
urlopen=build_opener(ProxyHandler({})).open


class DemoTests(unittest.TestCase):
    def setUp(self):
        temp=ROOT/'outputs/test-temp';temp.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=temp)
        self.open_data=[]
        self.root=Path(self.tmp.name);self.cache=self.root/'cache';self.run=self.root/'run'
        self.cache.mkdir();self.run.mkdir()
        rgb=np.zeros((3,6,8,3),np.uint8);rgb[...,0]=255
        np.save(self.cache/'frames.npy',rgb)
        (self.cache/'frames.csv').write_text('timestamp_s\n0\n0.1\n0.3\n')
        cm={'frames_sha256':file_hash(self.cache/'frames.npy'),'index_sha256':file_hash(self.cache/'frames.csv'),
            'clip_id':'synthetic_test','scope':'pipeline_check_not_final_evaluation'}
        (self.cache/'metadata.json').write_text(json.dumps(cm))
        depth=np.ones((3,6,8),np.float32)*16;depth[1,0,0]=np.nan
        np.save(self.run/'depths.npy',depth);np.save(self.run/'valid.npy',np.isfinite(depth))
        self.meta={'status':'complete','prediction_type':'metric_depth','unit':'m','alignment':'none','cache_metadata':cm,
            'model':{'model':'E4'},'config_status':'provisional',
            'depths_sha256':file_hash(self.run/'depths.npy'),'valid_sha256':file_hash(self.run/'valid.npy')}
        (self.run/'metadata.json').write_text(json.dumps(self.meta))

    def tearDown(self):
        for data in self.open_data:data.close()
        self.tmp.cleanup()

    def load_data(self):
        data=DemoData(self.cache,[self.run],depth_max=10)
        self.open_data.append(data)
        return data

    def test_readout_is_raw_z_and_invalid_depth_has_no_value(self):
        data=self.load_data()
        point=data.point(0,0)
        self.assertEqual(point['timestamps'],[0,.1,.3])
        self.assertEqual(point['traces'][0]['values'],[16,None,16])
        np.testing.assert_array_equal(data.depth_rgb(1,0)[0,0],[180,180,180])
        self.assertTrue(data.public_metadata()['methods'][0]['provisional'])
        with self.assertRaises(ValueError):data.point(8,0)

    def test_relative_output_is_rejected_in_metric_application(self):
        self.meta['prediction_type']='inverse_depth_like';self.meta['unit']='arbitrary'
        (self.run/'metadata.json').write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError,'metres'):
            DemoData(self.cache,[self.run])

    def test_loopback_http_serves_verified_pixels_and_rejects_arbitrary_paths(self):
        server=create_server(self.load_data(),ROOT/'demo',0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(url+'/api/meta') as response:meta=json.load(response)
            self.assertEqual(meta['frames'],3)
            with urlopen(url+'/api/point?x=0&y=0') as response:point=json.load(response)
            self.assertIsNone(point['traces'][0]['values'][1])
            with urlopen(url+'/api/frame?index=0&method=-1') as response:png=response.read()
            with Image.open(io.BytesIO(png)) as image:self.assertEqual(image.getpixel((0,0)),(255,0,0))
            for path,code in [('/api/point?x=-1&y=0',400),('/api/frame?index=3&method=0',400),('/../configs/paths.local.yaml',404)]:
                with self.subTest(path=path),self.assertRaises(HTTPError) as error:urlopen(url+path)
                self.assertEqual(error.exception.code,code)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':
    unittest.main()
