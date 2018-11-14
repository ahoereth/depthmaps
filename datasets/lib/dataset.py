import atexit
import os
import shutil
import random
from glob import glob
from pathlib import Path

import tensorflow as tf
from tensorflow.contrib.data import Dataset as TFDataset, Iterator

# Temporary directory next to the `datasets` folder.
tmpdir = (Path(__file__).parent / '..' / '..' / 'tmp')
os.makedirs(str(tmpdir), exist_ok=True)
DATA_DIR = tmpdir.resolve()


class Dataset:
    directory = DATA_DIR
    has_predefined_split = False
    input_shape = (0, 0)
    target_shape = (0, 0)
    test_only = False
    has_targets = True

    test_files = []
    train_files = []

    def __init__(self,
                 cleanup_on_exit=False,
                 use_predefined_split=False,
                 test_split=20,
                 workers=4,
                 checkpoint_dir=None):
        if cleanup_on_exit:
            atexit.register(self._cleanup)
        self.workers = workers

        d = self.directory

        testfiles_file = None
        if checkpoint_dir is not None:
            testfiles_file = Path(checkpoint_dir) / 'test_files.txt'
            if not os.path.isfile(str(testfiles_file)):
                testfiles_file = None
                print('Could not determine test_files from checkpoint dir.')

        if testfiles_file is not None:
            # Use testsplit as defined by checkpoint dir.
            with open(testfiles_file, 'r') as file:
                read = [(str(d / image), str(d / depth))
                        for pair in file.read().split('\n')
                        for image, depth in pair.split(',')]
            inputs = glob(str(d / '**/*.image.*'), recursive=True)
            targets = glob(str(d / '**/*.depth.*'), recursive=True)
            pairs = self._match_pairs(inputs, targets)
            self.test_files = [pair for pair in pairs if pair in read]
            self.train_files = [pair for pair in pairs if pair not in read]
        elif use_predefined_split and self.has_predefined_split:
            # Use original test/train split.
            inputs = glob(str(d / 'test/**/*.image.*'), recursive=True)
            targets = glob(str(d / 'test/**/*.depth.*'), recursive=True)
            self.test_files = self._match_pairs(inputs, targets)
            inputs = glob(str(d / 'train/**/*.image.*'), recursive=True)
            targets = glob(str(d / 'train/**/*.depth.*'), recursive=True)
            self.train_files = self._match_pairs(inputs, targets)
        else:
            # Create test/train split.
            inputs = glob(str(d / '**/*.image.*'), recursive=True)
            targets = glob(str(d / '**/*.depth.*'), recursive=True)
            pairs = self._match_pairs(inputs, targets)
            if self.test_only:
                self.test_files = pairs
                self.train_files = []
            else:
                n = len(pairs)
                perm = random.sample(range(n), n)
                self.train_files = [pairs[i] for i in perm[n // test_split:]]
                self.test_files = [pairs[i] for i in perm[:n // test_split]]

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

    def _get_feed(self, attrname, epochs=1):
        data = getattr(self, attrname)
        i, d = [tf.convert_to_tensor(x, tf.string) for x in list(zip(*data))]
        tfdataset = TFDataset.from_tensor_slices((i, d))
        tfdataset = tfdataset.shuffle(buffer_size=len(data[0]))
        tfdataset = tfdataset.map(
            self._parse_images,
            num_threads=self.workers,
            output_buffer_size=1000)
        tfdataset = tfdataset.batch(self.batchsize)
        tfdataset = tfdataset.repeat(epochs)
        iterator = tfdataset.make_one_shot_iterator()
        return iterator.string_handle()

    def create_test_feed(self, epochs=1):
        return self._get_feed('test_files', epochs)

    def create_train_feed(self, epochs=1):
        return self._get_feed('train_files', epochs)

    def _cleanup(self):
        """Delete temporary folders on exit."""
        shutil.rmtree(self.directory, ignore_errors=True)

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
        # Print targets left over.
        for targetfile in targets:
            path = Path(targetfile)
            print('No matching input found: {}'.format(path.name))
        return pairs

    def _parse_images(self, input_filepath, target_filepath):
        """Read input and target image from file and preprocess them."""
        input_shape, target_shape = self.output_shapes
        input_image = self._parse_image(input_filepath, input_shape)
        target_image = self._parse_image(target_filepath, target_shape)
        return input_image, target_image

    @classmethod
    def _parse_image(cls, filepath, shape):
        """Read image from file, resize it and scale its values from 0 to 1."""
        read = tf.read_file(filepath)
        decoded = tf.image.decode_png(read, channels=shape[-1], dtype=tf.uint8)
        scaled = tf.image.convert_image_dtype(decoded, tf.float32)
        resized = tf.image.resize_images(scaled, shape[:2])
        return resized

    def view(self):
        """Display samples from dataset using dataviewer."""
        from . import Dataviewer
        if not Dataviewer.AVAILABLE:
            raise RuntimeError('No GUI available.')
        data = self.test_files + self.train_files
        random.shuffle(data)
        print('Showing the complete dataset (test and train) in random order.')
        print('Dataset size: ', len(data))
        Dataviewer(
            data,
            name=self.__class__.__name__,
            keys=['image', 'depth'],
            cmaps={'depth': 'gray'})

    def __str__(self):
        """Creates a human readable test file csv."""
        sub = len(str(self.directory)) + 1
        return '\n'.join([
            ','.join([image[sub:], depth[sub:]])
            for image, depth in self.test_files
        ])
