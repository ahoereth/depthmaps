import atexit
import os
import shutil
import random
from glob import glob
from pathlib import Path

import tensorflow as tf
from tensorflow.contrib.data import Dataset as TFDataset, Iterator


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
        self.train_files = [pairs[i] for i in perm[len(pairs) // 10:]]
        self.test_files = [pairs[i] for i in perm[:len(pairs) // 10]]

    def finalize(self, shapes, batchsize):
        assert not hasattr(self, 'output_shapes')
        self.output_shapes = shapes
        self.batchsize = batchsize
        ishapes = [[None] + list(s) for s in self.output_shapes]
        itypes = (tf.float32, tf.float32)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = Iterator.from_string_handle(handle, itypes, tuple(ishapes))
        getter = iterator.get_next()
        return getter, handle

    def _get_feed(self, attrname, epochs=1, threads=4):
        data = getattr(self, attrname)
        inputs, targets = [tf.convert_to_tensor(x, tf.string)
                           for x in list(zip(*data))]
        tfdataset = TFDataset.from_tensor_slices((inputs, targets))
        tfdataset = tfdataset.shuffle(buffer_size=10000)
        tfdataset = tfdataset.map(self._parse_images, num_threads=threads,
                                  output_buffer_size=self.batchsize * 4)
        tfdataset = tfdataset.batch(self.batchsize)
        tfdataset = tfdataset.repeat(epochs)
        iterator = tfdataset.make_one_shot_iterator()
        return iterator.string_handle()

    def create_test_feed(self, epochs=1, threads=2):
        return self._get_feed('test_files', epochs, threads)

    def create_train_feed(self, epochs=1, threads=2):
        return self._get_feed('train_files', epochs, threads)

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

    def _parse_images(self, input_filepath, target_filepath):
        """Read input and target image from file and preprocess them."""
        input_shape, target_shape = self.output_shapes
        input_image = self._parse_image(input_filepath, input_shape)
        target_image = self._parse_image(target_filepath, target_shape)
        return input_image, target_image

    @classmethod
    def _parse_image(cls, filepath, shape):
        """Read image from file, resize it and scale its values from -1 to 1"""
        read = tf.read_file(filepath)
        decoded = tf.image.decode_png(read, channels=shape[-1])
        resized = tf.image.resize_images(decoded, shape[:2])
        scaled = (tf.image.convert_image_dtype(resized, tf.float32) - .5) * 2
        return scaled

    def view(self):
        """Display samples from dataset using dataviewer."""
        import numpy as np
        from PIL import Image
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not available.')
            return
        from . import Dataviewer

        samples = random.sample(self.train_files, k=100)
        images = [(Image.open(a), Image.open(b)) for a, b in samples]
        images = [(a, b.resize(a.size)) for a, b in images]
        Dataviewer(images, name=self.__class__.__name__,
                   keys=['image', 'depth'],
                   cmaps={'depth': 'gray'})
        plt.show()
