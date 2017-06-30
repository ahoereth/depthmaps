"""Downloads and extracts the Make3D depth image dataset.

https://cs.stanford.edu/group/reconstruction3d/Readme
"""
import os
from glob import glob
from pathlib import Path

from scipy import misc as spmisc, io as spio


from .dataset import extract, maybe_download
from .dataset import Dataset


FILES = {
    'train_data': 'http://cs.stanford.edu/group/reconstruction3d/Train400Img.tar.gz',
    'train_targets': 'http://cs.stanford.edu/group/reconstruction3d/Train400Depth.tgz',
    'test_data': 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134.tar.gz',
    'test_targets': 'http://www.cs.cornell.edu/~asaxena/learningdepth/Test134Depth.tar.gz',
}


class Make3D(Dataset):
    def __init__(self):
        super(Make3D, self).__init__()
        self.files = {}
        for name, url in FILES.items():
            archive = self.download(url)
            directory = extract(archive)
            self._tempdirs.append(directory)
            if name.endswith('data'):
                self.files[name] = glob(str(directory / '**' / '*.jpg'),
                                        recursive=True)
            elif name.endswith('targets'):
                files = []
                for mat in glob(str(directory / '**/*.mat'), recursive=True):
                    try:
                        depth = spio.loadmat(mat)['Position3DGrid'][..., 3]
                        img = spmisc.toimage(depth)
                        files.append(str(Path(mat).with_suffix('.jpg')))
                        img.save(files[-1], 'JPEG')
                        os.remove(mat)
                    except ValueError:
                        print("Couldn't open {}.".format(mat))
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
