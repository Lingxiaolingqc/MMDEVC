"""Real decode check: variable presentation timestamps must survive subsampling."""
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.data.video import decode_video, output_size


class VideoTests(unittest.TestCase):
    def test_variable_pts_and_rgb_order(self):
        import av
        base=ROOT/'outputs/test-temp'
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            path=Path(tmp)/'variable.mp4'
            with av.open(str(path),'w') as container:
                stream=container.add_stream('mpeg4',rate=30)
                stream.width,stream.height,stream.pix_fmt=64,48,'yuv420p'
                for pts in [0,1,3,6,9,12]:
                    rgb=np.zeros((48,64,3),dtype=np.uint8)
                    rgb[:,:,0]=255
                    frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
                    frame.pts,frame.time_base=pts,Fraction(1,30)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            frames,index,info=decode_video(path,target_fps=10,max_long_edge=64)
            self.assertEqual([x['source_frame_index'] for x in index],[0,2,3,4,5])
            np.testing.assert_allclose([x['timestamp_s'] for x in index],[0,.1,.2,.3,.4],atol=1e-6)
            self.assertEqual(frames.shape,(5,48,64,3))
            self.assertGreater(float(frames[:,:,:,0].mean()),245)
            self.assertLess(float(frames[:,:,:,2].mean()),5)
            self.assertAlmostEqual(info['effective_fps'],10)

    def test_no_upscale_and_even_encoder_dimensions(self):
        self.assertEqual(output_size(1920,1080,640),(640,360))
        self.assertEqual(output_size(320,240,640),(320,240))


if __name__=='__main__':
    unittest.main()
