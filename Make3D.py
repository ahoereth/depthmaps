"""Downloads and extracts the Make3D depth image dataset.
https://cs.stanford.edu/group/reconstruction3d/Readme

Usage:
```python
from Make3D import train_pairs, test_pairs

# Visualize some samples.
from matplotlib import pyplot as plt
from scipy import misc
for rgb, d in train_pairs[:10]:
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(rgb)
    ax2.imshow(misc.imresize(d, rgb.shape))
plt.show()

# Extract lists of data and target lists for training.
train_data, train_targets = zip(*train_pairs)
test_data, test_targets = zip(*test_pairs)
```
"""

import tarfile
import os
import shutil
from urllib import request
from glob import glob

import numpy as np
from scipy.io import loadmat
from scipy import misc as spmisc
from PIL import Image


TRAIN_DATA = 'http://cs.stanford.edu/group/reconstruction3d/Train400Img.tar.gz'
TRAIN_TARGETS = 'http://cs.stanford.edu/group/reconstruction3d/Train400Depth.tgz'
TEST_DATA = 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134.tar.gz'
TEST_TARGETS = 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134Depth.tar.gz'
DATA_DIRECTORY = 'data'
IMAGE_SHAPE = (2272, 1704)
IMAGE_SIZE = 55
NUM_CHANNELS = 3


def maybe_download(url):
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not os.path.exists(filepath):
        print('Downloading {}'.format(filename))
        filepath, _ = request.urlretrieve(url, filepath)
        print('Successfully downloaded {}'.format(filename))
    return filepath


def extract(filename, targets=False):
    ext = '.tar.gz' if not filename.endswith('.tgz') else '.tgz'
    orgdir = filename[:-len(ext)]
    thumbdir = orgdir + '_128'

    if os.path.exists(thumbdir):
        return glob(os.path.join(thumbdir, '*.jpg'))

    os.makedirs(thumbdir, exist_ok=True)

    print('Extracting {} to {}'.format(filename, orgdir))
    with tarfile.open(filename) as tar:
        tar.extractall(orgdir)

    result = []
    if targets:
        pattern = os.path.join(orgdir, '**', '*.mat')
        for infile in glob(pattern, recursive=True):
            noext, _ = os.path.splitext(os.path.basename(infile))
            outfile = os.path.join(thumbdir, noext + '.jpg')
            try:
                depth = loadmat(infile)
                grid = depth['Position3DGrid'][:, :, 3]
                img = spmisc.toimage(grid)
                img.save(outfile, 'JPEG')
            except ValueError:
                print('Couldn\'t open {}'.format(infile))
            else:
                result.append(outfile)
    else:
        print('Rescaling images from {} to {}'.format(orgdir, thumbdir))
        pattern = os.path.join(orgdir, '**', '*.jpg')
        for infile in glob(pattern, recursive=True):
            outfile = os.path.join(thumbdir, os.path.basename(infile))
            try:
                img = Image.open(infile)
                img.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                img.save(outfile, 'JPEG')
            except IOError:
                print('Couldn\'t rescale {}'.format(infile))
            else:
                result.append(outfile)

    print('Removing {} and not removing {}'.format(orgdir, filename))
    shutil.rmtree(orgdir)
    # os.remove(filename)
    return result


def matchpairs(data, targets, images=True):
    pairs = []
    for datafile in data:
        _, right = datafile.split('img')
        try:
            targetfile = next(t for t in targets if t.endswith(right))
            if images:
                rgb = spmisc.imread(datafile)
                d = spmisc.imread(targetfile)
                pairs.append((rgb, d))
            else:
                pairs.append((datafile, targetfile))
        except StopIteration:
            pass
    return pairs


train_data_filename = maybe_download(TRAIN_DATA)
train_targets_filename = maybe_download(TRAIN_TARGETS)
test_data_filename = maybe_download(TEST_DATA)
test_targets_filename = maybe_download(TEST_TARGETS)

train_data = extract(train_data_filename)
train_targets = extract(train_targets_filename, targets=True)
test_data = extract(test_data_filename)
test_targets = extract(test_targets_filename, targets=True)

train_pairs = matchpairs(train_data, train_targets)
test_pairs = matchpairs(test_data, test_targets)
