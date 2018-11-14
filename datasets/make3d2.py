"""Make3D-2 depth image dataset.

@article{Saxena2009,
  title = {Make3d: Learning 3d scene structure from a single still image},
  author = {Saxena, Ashutosh and Sun, Min and Ng, Andrew Y},
  journal = {IEEE transactions on pattern analysis and machine intelligence},
  volume = {31},
  number = {5},
  pages = {824--840},
  year = {2009},
  publisher = {IEEE}
}

@article{Saxena2008,
  title = {3-d depth reconstruction from a single still image},
  author = {Saxena, Ashutosh and Chung, Sung H and Ng, Andrew Y},
  journal = {International journal of computer vision},
  volume = {76},
  number = {1},
  pages = {53--69},
  year = {2008},
  publisher = {Springer}
}
"""
import os
from glob import glob
from pathlib import Path

from scipy import misc as spmisc, io as spio
from PIL import Image

from .lib import DATA_DIR, Dataset, maybe_extract, maybe_download

FILES = {
    'feature_data':
    'http://cs.stanford.edu/people/asaxena/learningdepth/Data/Dataset1_Images.tar.gz',
    'feature_targets':
    'http://cs.stanford.edu/people/asaxena/learningdepth/Data/Dataset1_Depths.tar.gz',
    'test_data':
    'http://cs.stanford.edu/people/asaxena/learningdepth/Data/Dataset2_Images.tar.gz',
    'test_targets':
    'http://www.cs.cornell.edu/~asaxena/learningdepth/Data/Dataset2_Depths.tar.gz',
    'train_data':
    'http://www.cs.cornell.edu/~asaxena/learningdepth/Data/Dataset3_Images.tar.gz',
    'train_targets':
    'http://cs.stanford.edu/people/asaxena/learningdepth/Data/Dataset3_Depths.tar.gz',
}


class Make3D2(Dataset):
    directory = DATA_DIR / 'make3d2'
    has_predefined_split = True
    input_shape = (480, 320)
    target_shape = (55 * 480 // 320, 55)

    def __init__(self, *args, **kwargs):
        for name, url in FILES.items():
            group = name.split('_')[0]
            archive, _ = maybe_download(self.directory / group, url)
            target_dir, extracted = maybe_extract(archive)
            if extracted:
                self._preprocess_data(name, target_dir)
        super().__init__(*args, **kwargs)

    def _preprocess_data(self, name, directory):
        """Preprocess a part of the 4 way split dataset."""
        if name.endswith('data'):
            for path in glob(str(directory / '**/*.jpg'), recursive=True):
                try:
                    with Image.open(path) as img:
                        if not name.startswith('feature'):
                            img = img.rotate(-90, 0, 1)
                        img = img.resize(self.input_shape)
                except (ValueError, OSError):
                    print("Couldn't open {}".format(path))
                else:
                    path = Path(path)
                    filename = path.name.split('img-')[1]
                    target = (path.parent / filename).with_suffix('.image.png')
                    img.save(target, 'PNG')
                os.remove(str(path))
        elif name.endswith('targets'):
            for path in glob(str(directory / '**/*.mat'), recursive=True):
                try:
                    mat = spio.loadmat(path)['depthMap']
                    img = spmisc.toimage(mat).resize(self.target_shape)
                except ValueError:
                    print("Couldn't open {}".format(path))
                else:
                    path = Path(path)
                    name = path.name[path.name.index('-') + 1:]
                    target = (path.parent / name).with_suffix('.depth.png')
                    img.save(target, 'PNG')
                os.remove(str(path))


if __name__ == '__main__':
    """Take a look at the data using `python -m datasets.make3d2`."""
    Make3D2().view()
