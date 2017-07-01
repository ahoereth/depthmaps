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

from .utils import maybe_extract, maybe_download
from .dataset import Dataset, datadirectory


FILES = {
    'train_data': 'http://cs.stanford.edu/group/reconstruction3d/Train400Img.tar.gz',
    'train_targets': 'http://cs.stanford.edu/group/reconstruction3d/Train400Depth.tgz',
    'test_data': 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134.tar.gz',
    'test_targets': 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134Depth.tar.gz',
}


class Make3D(Dataset):
    input_shape = (2272 // 3, 1704 // 3)
    target_shape = (55, 305)

    def __init__(self, cleanup_on_exit=False):
        super(Make3D, self).__init__(cleanup_on_exit=cleanup_on_exit)
        for name, url in FILES.items():
            archive, _ = maybe_download(datadirectory, url)
            directory, extracted = maybe_extract(archive)
            self._tempdirs.append(directory)
            if not extracted:
                continue
            if name.endswith('data'):
                files = []
                for path in glob(str(directory / '**/*.jpg'), recursive=True):
                    try:
                        with Image.open(path) as img:
                            img = img.resize(self.input_shape)
                            img.save(path)
                    except (ValueError, OSError):
                        print("Couldn't open {}.".format(path))
                    else:
                        files.append(path)
                self.files[name] = files
            elif name.endswith('targets'):
                files = []
                for path in glob(str(directory / '**/*.mat'), recursive=True):
                    try:
                        mat = spio.loadmat(path)['Position3DGrid'][..., 3]
                        img = spmisc.toimage(mat)
                    except ValueError:
                        print("Couldn't open {}.".format(path))
                    else:
                        files.append(str(Path(path).with_suffix('.jpg')))
                        img.save(files[-1], 'JPEG')
                    os.remove(path)
                self.files[name] = files

        self.train_files = self.match_pairs(self.files['train_data'],
                                            self.files['train_targets'])
        self.test_files = self.match_pairs(self.files['test_data'],
                                           self.files['test_targets'])

    def match_pairs(self, data, targets):
        pairs = []
        for datafile in data:
            _, identifier = datafile.split('img')
            try:
                targetfile = next(t for t in targets if t.endswith(identifier))
                pairs.append((datafile, targetfile))
            except StopIteration:
                pass
        return pairs
