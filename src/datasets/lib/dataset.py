import atexit
import os
import shutil
import random
from glob import glob
from pathlib import Path

import tensorflow as tf
from tensorflow.contrib.data import Dataset as TFDataset


DATA_DIR = Path.cwd() / 'tmp'


class Dataset:
    directory = DATA_DIR
    input_shape = (0, 0)
    target_shape = (0, 0)

    _tempdirs = []

    def __init__(self, cleanup_on_exit=False):
        if cleanup_on_exit:
            atexit.register(self._cleanup)

        # Create test/train split.
        inputs = glob(str(self.directory / '**/*.image.*'), recursive=True)
        targets = glob(str(self.directory / '**/*.depth.*'), recursive=True)
        pairs = self._match_pairs(inputs, targets)
        perm = random.sample(range(len(pairs)), len(pairs))
        train_files = [pairs[i] for i in perm[:len(pairs) // 10]]
        test_files = [pairs[i] for i in perm[len(pairs) // 10:]]

        inputs, targets = zip(*train_files)
        tfdataset = TFDataset.from_tensor_slices((inputs, targets))
        tfdataset = tfdataset.shuffle()
        self.train = tfdataset.map(self._parse_files)

        inputs, targets = zip(*test_files)
        tfdataset = TFDataset.from_tensor_slices((inputs, targets))
        tfdataset = tfdataset.shuffle()
        self.test = tfdataset.map(self._parse_files)

    def __call__(self):
        return self.train, self.test

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

    def _parse_files(self, input_filepath, target_filepath):
        input_image = tf.image.decode_image(tf.read_file(input_filepath))
        target_image = tf.image.decode_image(tf.read_file(target_filepath))
        return input_image, target_image
