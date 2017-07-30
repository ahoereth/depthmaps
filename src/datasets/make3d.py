"""Downloads and extracts the Make3D depth image dataset.

https://cs.stanford.edu/group/reconstruction3d/Readme

@inproceedings{Saxena2009,
    author = "Ashutosh Saxena and Min Sun and Andrew Y. Ng",
    title = {Make3D: Learning 3D Scene Structure from a Single Still Image},
    journal = {IEEE PAMI},
    volume = {30},
    number = {5},
    pages = {824--840},
    year = {2009}
}
"""
import os
from glob import glob
from pathlib import Path

from scipy import misc as spmisc, io as spio
from PIL import Image

from .lib import DATA_DIR, Dataset, maybe_extract, maybe_download


FILES = {
    'train_data': 'http://cs.stanford.edu/group/reconstruction3d/Train400Img.tar.gz',
    'train_targets': 'http://cs.stanford.edu/group/reconstruction3d/Train400Depth.tgz',
    'test_data': 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134.tar.gz',
    'test_targets': 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134Depth.tar.gz',
}


class Make3D(Dataset):
    input_shape = (480, 320)
    target_shape = (55 * 480 // 320, 55)
    # input_shape = (2272, 1704)
    # target_shape = (55, 305)

    def __init__(self, cleanup_on_exit=False):
        super(Make3D, self).__init__(cleanup_on_exit=cleanup_on_exit)
        directory = DATA_DIR / 'make3d'
        for name, url in FILES.items():
            archive, _ = maybe_download(directory, url)
            target_dir, extracted = maybe_extract(archive)
            self._tempdirs.append(target_dir)
            if extracted:
                self._preprocess_data(name, target_dir)

        # Create train/test split.
        self._split(directory)

    def _preprocess_data(self, name, directory):
        """Preprocess a part of the 4 way split dataset."""
        if name.endswith('data'):
            for path in glob(str(directory / '**/*.jpg'), recursive=True):
                try:
                    with Image.open(path) as img:
                        img = img.resize(self.input_shape)
                except (ValueError, OSError):
                    print("Couldn't open {}.".format(path))
                else:
                    path = Path(path)
                    name = path.name.split('img-')[1]
                    target = (path.parent / name).with_suffix('.image.jpg')
                    img.save(target, 'JPEG')
                os.remove(str(path))
        elif name.endswith('targets'):
            for path in glob(str(directory / '**/*.mat'), recursive=True):
                try:
                    mat = spio.loadmat(path)['Position3DGrid'][..., 3]
                    img = spmisc.toimage(mat)
                except ValueError:
                    print("Couldn't open {}.".format(path))
                else:
                    path = Path(path)
                    name = path.name.split('depth_sph_corr-')[1]
                    target = (path.parent / name).with_suffix('.depth.jpg')
                    img.save(target, 'JPEG')
                os.remove(str(path))
