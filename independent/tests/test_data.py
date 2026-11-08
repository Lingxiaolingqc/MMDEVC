import h5py
import numpy as np
import pytest
from scipy.io import savemat
from depth_project import data


def test_hdf5_axes_and_filled_depth(tmp_path):
    path=tmp_path/'nyu.mat'
    with h5py.File(path,'w') as f:
        image=np.zeros((1,3,3,2),np.uint8); image[0,:,2,1]=[10,20,30]
        depth=np.zeros((1,3,2),np.float32); depth[0,2,1]=4.5
        f['images']=image; f['depths']=depth
    rgb,depth=data.load_nyu_sample(0,path)
    assert rgb.shape==(2,3,3)
    np.testing.assert_array_equal(rgb[1,2],[10,20,30])
    assert depth.shape==(2,3)
    assert depth[1,2]==4.5


def test_matlab_split_indices_become_zero_based(tmp_path):
    path=tmp_path/'splits.mat'
    savemat(path,{'trainNdxs':np.array([[1],[3]]),'testNdxs':np.array([[2]])})
    split=data.load_nyu_split(path)
    np.testing.assert_array_equal(split['development'],[0,2])
    np.testing.assert_array_equal(split['test'],[1])


@pytest.mark.parametrize('little',[True,False])
def test_pfm_endianness_and_bottom_up_rows(tmp_path,little):
    path=tmp_path/'depth.pfm'
    order='<f4' if little else '>f4'
    path.write_bytes(b'Pf\n2 2\n'+(b'-1.0\n' if little else b'1.0\n')+np.array([[3,4],[1,2]],dtype=order).tobytes())
    np.testing.assert_array_equal(data.read_pfm(path),[[1,2],[3,4]])


def test_middlebury_pfm_scale_does_not_rescale_disparity_pixels(tmp_path):
    path=tmp_path/'disparity.pfm'
    path.write_bytes(b'Pf\n2 1\n-0.003922\n'+np.array([10,20],dtype='<f4').tobytes())
    np.testing.assert_array_equal(data.read_pfm(path),[[10,20]])
