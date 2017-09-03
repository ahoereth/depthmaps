#import scipy.io as sio
import h5py
import numpy as np

data = h5py.File("data/nyu_depth_v2_labeled.mat")

img = np.array(data['images'])
img = np.swapaxes(img, 1,3)
depths = np.array(data['depths'])
depths = np.swapaxes(depths, 1,2)

l = list(zip(img, depths))

def nyu_data(train_test_ratio = .8):
    cut = (int(.8*len(l)))
    train_pairs = l[:cut]
    test_pairs = l[cut:]
    return train_pairs, test_pairs