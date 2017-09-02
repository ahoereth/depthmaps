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

        self.step = tf.train.get_or_create_global_step()

        self.training = tf.placeholder_with_default(False, None)

        # Resize and scale the test and train data, provide iterators.
        shapes = (self.input_shape, self.target_shape)
        feed, self.feedhandle = self.dataset.finalize(shapes, self.batchsize)
        self.inputs, self.targets = feed

        # Create the network.
        self.net = self.build_network(self.inputs, self.targets,
                                      training=self.training)

        self.summaries = tf.summary.merge_all()

        # Keep moving averages for the loss.
        # train_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # test_ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # train_ema_op = train_ema.apply(to_tuple(self.net.loss))
        # test_ema_op = test_ema.apply(to_tuple(self.test_net.loss))

    def build_network(self, inputs, targets, training=False):
        """Create the neural network."""
        raise NotImplementedError

    def _train(self, sess, handle):
        """Train the model."""
        while not sess.should_stop():
            try:
                sess.run(self.net.train, {self.feedhandle: handle,
                                          self.training: True})
            except tf.errors.OutOfRangeError:
                break

    def train(self, epochs=1, workers=2):
        handle_op = self.dataset.create_train_feed(epochs)

        summary_hook = tf.train.SummarySaverHook(output_dir=str(self.logdir),
                                                 summary_op=self.summaries,
                                                 save_secs=20)
        step_hook = tf.train.StepCounterHook(output_dir=str(self.logdir),
                                             every_n_steps=100)
        hooks = [summary_hook, step_hook]

        kwargs = dict(checkpoint_dir=str(self.logdir), hooks=hooks)
        with tf.train.SingularMonitoredSession(**kwargs) as sess:
            handle = sess.raw_session().run(handle_op)
            threads = [Thread(target=self._train, args=(sess, handle))
                       for _ in range(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def evaluate(self, fetch_images=True):
        """Evaluate the model.

        Still passes the data through the model in the specified batchsize
        in order to prevent out of memory errors. Basically performs a whole
        epoch of feed forward steps and collects the results.
        """
        assert tf.train.checkpoint_exists(str(self.logdir)), \
            'No checkpoint found in logdir: {}'.format(self.logdir)

        results = []
        handle_op = self.dataset.create_train_feed()
        kwargs = dict(checkpoint_dir=str(self.logdir))
        with tf.train.SingularMonitoredSession(**kwargs) as sess:
            handle = sess.raw_session().run(handle_op)
            while not sess.should_stop():
                if fetch_images:
                    inputs, targets, outputs = sess.run(
                        [self.inputs, self.targets, self.net.output],
                        {self.feedhandle: handle})
                    results.extend(list(zip(inputs, targets, outputs)))
                else:
                    sess.run(self.net.output, {self.feedhandle: handle})
        return results
