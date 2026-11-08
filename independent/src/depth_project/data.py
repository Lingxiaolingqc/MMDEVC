"""Official dataset readers."""
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def load_nyu_sample(index: int, path: Path = Path("data/nyu/nyu_depth_v2_labeled.mat")):
    with h5py.File(path, "r") as source:
        rgb = source["images"][index].transpose(2,1,0)
        depth = source["depths"][index].T
    return np.ascontiguousarray(rgb), np.asarray(depth, dtype=np.float32)


def load_nyu_split(path: Path):
    source = loadmat(path)
    result = {name:np.asarray(source[key],dtype=np.int64).reshape(-1)-1 for name,key in [("development","trainNdxs"),("test","testNdxs")]}
    if any((values < 0).any() or len(np.unique(values)) != len(values) for values in result.values()):
        raise ValueError("NYU split contains invalid/duplicate MATLAB indices")
    if np.intersect1d(result["development"],result["test"]).size:
        raise ValueError("NYU development and test split overlap")
    return result


def read_pfm(path: Path) -> np.ndarray:
    with open(path,"rb") as source:
        header=source.readline().strip()
        if header not in {b'Pf',b'PF'}:
            raise ValueError("Invalid PFM header")
        line=source.readline()
        while line.startswith(b'#'):
            line=source.readline()
        width,height=map(int,line.split())
        scale=float(source.readline())
        dtype='<f4' if scale < 0 else '>f4'
        values=np.frombuffer(source.read(),dtype=dtype)
    channels=3 if header==b'PF' else 1
    if len(values)!=width*height*channels:
        raise ValueError("Truncated PFM data")
    shape=(height,width,3) if channels==3 else (height,width)
    # Middlebury stores disparity in pixel-valued samples. Its -0.003922
    # header scale is an image-display convention, not a disparity unit.
    return np.flipud(values.reshape(shape)).astype(np.float32)


def read_calibration(path: Path) -> dict:
    result={}
    for line in Path(path).read_text().splitlines():
        key,value=line.split('=',1)
        if key.startswith('cam'):
            result[key]=np.array([[float(x) for x in row.split()] for row in value.strip('[]').split(';')])
        else:
            result[key]=float(value)
    return result
