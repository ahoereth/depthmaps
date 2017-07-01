import atexit
import os
import shutil
import random
from glob import glob
from pathlib import Path


class Dataset:
    input_shape = (0, 0)
    target_shape = (0, 0)

    def __init__(self, cleanup_on_exit=False):
        self.train_files = []
        self.test_files = []
        self._tempdirs = []
        if cleanup_on_exit:
            atexit.register(self._cleanup)

    def _split(self, directory):
        """Create test/train split."""
        inputs = glob(str(directory / '**/*.image.*'), recursive=True)
        targets = glob(str(directory / '**/*.depth.*'), recursive=True)
        pairs = self._match_pairs(inputs, targets)
        perm = random.sample(range(len(pairs)), len(pairs))
        self.train_files = [pairs[i] for i in perm[:len(pairs) // 10]]
        self.test_files = [pairs[i] for i in perm[len(pairs) // 10:]]

    def _cleanup(self):
        """Delete temporary folders on exit."""
        for path in self._tempdirs:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _match_pairs(inputs, targets):
        """Match input images to target images."""
        pairs = []
        targets = targets.copy()
        for inputfile in inputs:
            path = Path(inputfile)
            ext = path.suffix
            identifier = path.name.replace('.image' + ext, '.depth' + ext)
            try:
                i, targetfile = next((i, t) for i, t in enumerate(targets)
                                     if t.endswith(identifier))
                del targets[i]
                pairs.append((inputfile, targetfile))
            except StopIteration:
                print('No matching target found: {}'.format(path.name))
        return pairs
