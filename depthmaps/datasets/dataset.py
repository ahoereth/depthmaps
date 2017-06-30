import atexit
import os
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.error import HTTPError
from urllib import request

from scipy import misc as spmisc


DATADIRECTORY = str(Path(os.path.dirname(__file__)) / 'tmp')


def maybe_download(target: str, url: str, force=False):
    """Download a file if it does not exist at the specified target."""
    def progress(count, size, total):
        total = total / (1024 * 1024)
        size = (size / (1024 * 1024)) * count
        sys.stdout.write('\rProgress: {:.2f}/{:.2f}MB...'.format(size, total))

    target_path = Path(target)
    if target_path.suffix:
        target_dir = target_path.parent
    else:
        filename = os.path.basename(url)
        target_dir = target_path
        target_path = target_dir / filename

    os.makedirs(target_dir, exist_ok=True)

    if not target_path.exists() or force:
        print('Fetching `{}`...'.format(target_path))
        try:
            request.urlretrieve(url, str(target_path), progress)
        except HTTPError:
            print('Failed to download `{}`.'.format(url))
    else:
        print('`{}` already exists.'.format(target_path))

    return target_path


def extract(filename, target_dir=None, targets=False):
    """Extract a file to the specified target directory."""
    file = Path(filename)
    if target_dir is None:
        target_dir = file.parent / file.name[:-len(''.join(file.suffixes))]
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(file) as tar:
        tar.extractall(target_dir)
    return target_dir


class Dataset:
    def __init__(self):
        self.test_files = []
        self.train_files = []
        self._tempdirs = []
        atexit.register(self._cleanup)

    @property
    def test_images(self):
        if not hasattr(self, '_test_images'):
            self._test_images = [(spmisc.imread(data), spmisc.imread(target))
                                 for data, target in self.test_files]
        return self._test_images

    @property
    def train_images(self):
        if not hasattr(self, '_test_images'):
            self._train_images = [(spmisc.imread(data), spmisc.imread(target))
                                  for data, target in self.train_files]
        return self._train_images

    def _cleanup(self):
        for path in self._tempdirs:
            shutil.rmtree(path, ignore_errors=True)

    def download(self, url: str):
        return maybe_download(DATADIRECTORY, url)
