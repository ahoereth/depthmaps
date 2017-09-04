"""NYU depth image dataset.

@inproceedings{Silberman2012,
    author = "Nathan Silberman and Derek Hoiem and Pushmeet Kohli and Rob Fergus",
    title = {Indoor Segmentation and Support Inference from RGBD Images},
    booktitle = {ECCV},
    year = {2012}
}
"""
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import misc as spmisc

from .lib import DATA_DIR, Dataset, maybe_extract, maybe_download


URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'


class Nyu(Dataset):
    directory = DATA_DIR / 'nyu'
    has_predefined_split = False
    input_shape = (480, 320)
    target_shape = (55 * 480 // 320, 55)

    def __init__(self, *args, **kwargs):
        path, _ = maybe_download(DATA_DIR, URL)
        if not self.directory.exists():
            self._extract_mat(self.directory, path)
        super().__init__(*args, **kwargs)

    def _extract_mat(self, target_dir, mat_path):
        """Extract input and target images from mat file."""
        os.makedirs(str(target_dir), exist_ok=True)
        with h5py.File(mat_path) as mat:
            names = mat['rawRgbFilenames'][0]
            images = np.swapaxes(mat['images'], 1, 3)
            depths = np.swapaxes(mat['depths'], 1, 2)
            for name, image, depth in zip(names, images, depths):
                # Filename extraction, props @shoeffner
                name = (Path(''.join(map(chr, mat[name][:].T[0])))
                        .stem.replace('.', '_'))
                target_path = target_dir / name
                os.makedirs(str(target_path.parent), exist_ok=True)
                spmisc.toimage(image).resize(self.input_shape) \
                    .save(target_path.with_suffix('.image.png'))
                spmisc.toimage(depth).resize(self.target_shape) \
                    .save(target_path.with_suffix('.depth.png'))


if __name__ == '__main__':
    """Take a look at the data using `python -m datasets.nyu`."""
    Nyu().view()
