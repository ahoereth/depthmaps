"""Depth map generation base model."""
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from threading import Thread

import tensorflow as tf

from .utils import to_float


Network = namedtuple('Network', ('output', 'train', 'loss'))


class Model:
    """Depth map generation base model."""
    input_shape = (0, 0, 0)
    target_shape = (0, 0, 0)
    batchsize = 32

    def __init__(self, dataset, checkpoint=None):
        time = datetime.now().strftime('%y%m%d-%H%M')
        self.logdir = Path('logs') / type(self).__name__ / time

        self.dataset = dataset
        self.session = tf.Session()

        self.step = tf.train.create_global_step()

        self.training = tf.placeholder_with_default(False, None)

        # Resize and scale the test and train data, provide iterators.
        shapes = (self.input_shape, self.target_shape)
        feed, self.feedhandle = self.dataset.finalize(shapes, self.batchsize)
        self.inputs, self.targets = feed

        # Create the network.
        self.net = self.build_network(self.inputs, self.targets,
                                      training=self.training)

        # Keep moving averages for the loss.
        # train_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # test_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # train_ema_op = train_ema.apply(to_tuple(self.train_net.loss))
        # test_ema_op = test_ema.apply(to_tuple(self.test_net.loss))

        # Save the model regularly and maybe restore a saved model.
        self.saver = tf.train.Saver(max_to_keep=1,
                                    keep_checkpoint_every_n_hours=1)
        if checkpoint is not None:
            self.saver.restore(checkpoint)
        else:
            self.session.run(tf.global_variables_initializer())

        # Store summaries for tensorboard.
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(str(Path('logs') / self.logdir),
                                            self.session.graph)

    def build_network(self, inputs, targets, reuse=None):
        """Create the neural network."""
        raise NotImplementedError

    def _train(self, coord, handle, save_frequency=1000):
        """Train the model."""
        while not coord.should_stop():
            try:
                self.session.run([self.step, self.net.train],
                                 {self.feedhandle: handle,
                                  self.training: True})
            except tf.errors.OutOfRangeError:
                coord.request_stop()

    def train(self, epochs=1, workers=2, save_frequency=1000):
        handle = self.dataset.create_train_feed(self.session, epochs)
        coord = tf.train.Coordinator()
        kwargs = {'coord': coord, 'save_frequency': save_frequency,
                  'handle': handle}
        threads = [Thread(target=self._train, kwargs=kwargs)
                   for _ in range(workers)]
        for thread in threads:
            thread.start()
        coord.join(threads)

    def evaluate(self):
        """Evaluate the model.

        Still passes the data through the model in the specified batchsize
        in order to prevent out of memory errors. Basically performs a whole
        epoch of feed forward steps and collects the results.
        """
        results = []
        handle = self.dataset.create_train_feed(self.session)
        while True:
            try:
                inputs, targets, outputs = self.session.run(
                    [self.inputs, self.targets, self.net.output],
                    {self.feedhandle: handle})
            except tf.errors.OutOfRangeError:
                break
            results.extend(list(zip(inputs, targets, outputs)))
        return results
