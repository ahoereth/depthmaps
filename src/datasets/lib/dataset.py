import atexit
import os
import shutil
import random
from glob import glob
from pathlib import Path
from functools import partial

import tensorflow as tf
from tensorflow.contrib.data import Dataset as TFDataset


DATA_DIR = Path.cwd() / 'tmp'


to_string_tensor = partial(tf.convert_to_tensor, dtype=tf.string)


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
        self.train_files = [pairs[i] for i in perm[:len(pairs) // 10]]
        self.test_files = [pairs[i] for i in perm[len(pairs) // 10:]]

    def get(self, attrname, batchsize, shapes):
        data = getattr(self, attrname)
        inputs, targets = list(map(to_string_tensor, zip(*data)))
        tfdataset = TFDataset.from_tensor_slices((inputs, targets))
        tfdataset = tfdataset.shuffle(buffer_size=10000)
        tfdataset = tfdataset.map(partial(self._parse_images, shapes=shapes))
        tfdataset = tfdataset.batch(batchsize)
        return tfdataset

    def test(self, batchsize, shape):
        tfdataset = self.get('test_files', batchsize, shape)
        tfdataset = tfdataset.repeat()
        return tfdataset.make_initializable_iterator()

    def train(self, epochs, batchsize, shape):
        tfdataset = self.get('train_files', batchsize, shape)
        tfdataset = tfdataset.repeat(epochs)
        return tfdataset.make_one_shot_iterator()

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

    @classmethod
    def _parse_image(cls, filepath, shape):
        """Read image from file, resize it and scale its values from -1 to 1"""
        read = tf.read_file(filepath)
        decoded = tf.image.decode_png(read, channels=shape[-1])
        resized = tf.image.resize_images(decoded, shape[:2])
        scaled = (tf.image.convert_image_dtype(resized, tf.float32) - .5) * 2
        return scaled

    @classmethod
    def _parse_images(cls, input_filepath, target_filepath, shapes):
        """Read input and target image from file and preprocess them."""
        input_shape, target_shape = shapes
        input_image = cls._parse_image(input_filepath, input_shape)
        target_image = cls._parse_image(target_filepath, target_shape)
        return input_image, target_image
