"""Reproducible experiment orchestration."""
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from .data import read_calibration,read_pfm
from .metrics import stereo_metrics
from .records import sha256_file
from .stereo import compute_sgbm,disparity_to_meters


def run_stereo(data_root: Path, output: Path, expected_pairs: int = 15) -> dict:
    data_root=Path(data_root); output=Path(output)
    pairs=sorted(p for p in data_root.glob('*') if p.is_dir() and (p/'im0.png').is_file())
    if len(pairs)!=expected_pairs:
        raise ValueError(f"Expected {expected_pairs} stereo pairs, found {len(pairs)} in {data_root}")
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pair in pairs:
        calibration=read_calibration(pair/'calib.txt')
        left=cv2.imread(str(pair/'im0.png'),cv2.IMREAD_GRAYSCALE)
        right=cv2.imread(str(pair/'im1.png'),cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise ValueError(f"Missing/corrupt stereo image in {pair}")
        disparities=int(np.ceil(calibration['ndisp']/16))*16
        start=time.perf_counter()
        prediction=compute_sgbm(left,right,num_disparities=disparities)
        elapsed=time.perf_counter()-start
        depth=disparity_to_meters(prediction,calibration['cam0'][0,0],calibration['baseline'],calibration['doffs'])
        np.savez_compressed(output/f'{pair.name}.npz',disparity_px=prediction,depth_m=depth.astype(np.float32),valid=np.isfinite(depth))
        row={'pair':pair.name,**stereo_metrics(prediction,read_pfm(pair/'disp0GT.pfm')),'inference_s':elapsed,'num_disparities':disparities,'calibration_sha256':sha256_file(pair/'calib.txt'),'gt_sha256':sha256_file(pair/'disp0GT.pfm')}
        rows.append(row)
        print(f"{pair.name}: MAE={row['mae_px']:.4f}px coverage={row['coverage']:.2%}",flush=True)
    summary={'pairs':rows,'mean_pair_mae_px':float(np.mean([row['mae_px'] for row in rows])),'mean_pair_coverage':float(np.mean([row['coverage'] for row in rows])),'pixel_weighted_coverage':sum(row['prediction_valid_count'] for row in rows)/sum(row['gt_valid_count'] for row in rows),'mean_inference_s':float(np.mean([row['inference_s'] for row in rows])),'invalid_prediction_rule':'nonfinite or disparity <= 0 px','aggregation':'arithmetic mean of per-pair MAE and coverage; counts also reported','configuration':{'num_disparities':'ceil(calib.ndisp / 16) * 16','block_size':5,'P1':200,'P2':800,'uniquenessRatio':10,'speckleWindowSize':100,'speckleRange':2,'disp12MaxDiff':1,'preFilterCap':31,'mode':'SGBM_3WAY'},'hardware':platform.platform(),'opencv_version':cv2.__version__,'model_training':'none','selection':'fixed independent defaults; no GT parameter tuning'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary
