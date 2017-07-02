import random
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf
from scipy import misc as spmisc


class Dataflow:
    def __init__(self, session, filepath_tuples, shapes, batchsize, workers=1):
        self.session = session
        self.files = filepath_tuples
        self.shapes = shapes
        self._cache = {}

        self.inputs = [tf.placeholder(tf.uint8, shape) for shape in shapes]

        self.queue = tf.FIFOQueue(capacity=batchsize * 4,
                                  dtypes=[tf.uint8 for _ in shapes],
                                  shapes=[shape for shape in shapes])
        self.enqueue = self.queue.enqueue(self.inputs)
        self.out = self.queue.dequeue_many(batchsize)

        # Using a thread safe queue to ensure epoch-style feeding.
        self.index_queue = Queue(len(self.files))
        Thread(target=self.queue_worker, daemon=True).start()

        # Because of the index queue we can now use multiple workers to feed
        # data into the TensorFlow graph.
        for _ in range(workers):
            Thread(target=self.feed_worker, daemon=True).start()

    def load_file(self, filepath, shape):
        """Read image from filesystem and resize it to given shape."""
        flat = len(shape) == 2 or shape[2] == 1
        image = spmisc.imread(filepath, flat)
        image = spmisc.imresize(image, shape[:2])
        if len(shape) == 3 and len(image.shape) == 2:
            image = image[..., None]
        return image

    def get_images(self, index):
        """Either load images from the filesystem or the local cache dict.

        Idea is to only preprocess each image once and not hit the filesystem
        too often. Might become problematic with really big datasets.
        """
        images = []
        for filepath, shape in zip(self.files[index], self.shapes):
            try:
                images.append(self._cache[filepath])
            except KeyError:
                self._cache[filepath] = self.load_file(filepath, shape)
                images.append(self._cache[filepath])
        return images

    def queue_worker(self):
        """Enqueue epoch indices in random order into the index queue."""
        n = len(self.files)
        while True:
            for index in random.sample(range(n), n):
                self.index_queue.put(index)

    def feed_worker(self):
        """Dequeue from the index queue and feed images into TensorFlow."""
        while True:
            images = self.get_images(self.index_queue.get())
            self.session.run(self.enqueue, dict(zip(self.inputs, images)))
