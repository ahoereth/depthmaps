"""Make3D depth image dataset.

https://cs.stanford.edu/group/reconstruction3d/Readme

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

@article{Liu2016,
  title = {Learning depth from single monocular images using deep convolutional
           neural fields},
  author = {Liu, Fayao and Shen, Chunhua and Lin, Guosheng and Reid, Ian},
  journal = {IEEE transactions on pattern analysis and machine intelligence},
  volume = {38},
  number = {10},
  pages = {2024--2039},
  year = {2016},
  publisher = {IEEE}
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
    directory = DATA_DIR / 'make3d'
    predefined_split_available = True
    input_shape = (480, 320)
    target_shape = (55 * 480 // 320, 55)
    # input_shape = (2272, 1704)
    # target_shape = (55, 305)

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
                        img = img.resize(self.input_shape)
                except (ValueError, OSError):
                    print("Couldn't open {}".format(path))
                else:
                    path = Path(path)
                    filename = path.name.split('img-')[1]
                    if filename.startswith('10.21'):
                        filename = filename[5:]
                    target = (path.parent / filename).with_suffix('.image.png')
                    img.save(target, 'PNG')
                os.remove(str(path))
        elif name.endswith('targets'):
            for path in glob(str(directory / '**/*.mat'), recursive=True):
                try:
                    mat = spio.loadmat(path)['Position3DGrid'][..., 3]
                    img = spmisc.toimage(mat).resize(self.target_shape)
                except ValueError:
                    print("Couldn't open {}".format(path))
                else:
                    path = Path(path)
                    name = path.name.split('depth_sph_corr-')[1]
                    if name.startswith('10.21'):
                        name = name[5:]
                    target = (path.parent / name).with_suffix('.depth.png')
                    img.save(target, 'PNG')
                os.remove(str(path))


if __name__ == '__main__':
    """Take a look at the data using `python -m datasets.make3d`."""
    Make3D().view()
