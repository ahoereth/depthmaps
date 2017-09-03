"""Depth map generation base model."""
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from threading import Thread

import tensorflow as tf

from .feed_summary_saver_hook import FeedSummarySaverHook
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
        self.net = self.build_network(self.inputs, self.targets, self.training)

        tf.summary.image('inputs', self.inputs)
        tf.summary.image('targets', self.targets)
        tf.summary.image('outputs', self.net.output)
        self.summaries = tf.summary.merge_all()

    def build_network(self, inputs, targets, training):
        """Create the neural network."""
        raise NotImplementedError

    def train(self, epochs=1):
        handle_op = self.dataset.create_train_feed(epochs)
        test_handle_op = self.dataset.create_test_feed(epochs=-1)

        test_logs = str(self.logdir / 'test')
        train_logs = str(self.logdir / 'train')

        saver = tf.train.Saver(max_to_keep=24, keep_checkpoint_every_n_hours=1)
        checker = tf.train.CheckpointSaverHook(checkpoint_dir=test_logs,
                                               save_secs=60 * 60 * 10,
                                               saver=saver)
        summarizer = tf.train.SummarySaverHook(output_dir=train_logs,
                                               summary_op=self.summaries,
                                               save_steps=100)
        tester = FeedSummarySaverHook({self.feedhandle: test_handle_op},
                                      output_dir=test_logs,
                                      summary_op=self.summaries,
                                      save_steps=100)
        timer = tf.train.StepCounterHook(output_dir=train_logs)
        hooks = [checker, summarizer, timer, tester]

        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = \
            tf.OptimizerOptions.ON_2

        kwargs = dict(checkpoint_dir=test_logs, hooks=hooks,
                      config=config, stop_grace_period_secs=10)
        with tf.train.SingularMonitoredSession(**kwargs) as sess:
            handle = sess.raw_session().run(handle_op)
            while not sess.should_stop():
                sess.run(self.net.train, {self.feedhandle: handle,
                                          self.training: True})

    def evaluate(self, fetch_images=True):
        """Evaluate the model.

        Still passes the data through the model in the specified batchsize
        in order to prevent out of memory errors. Basically performs a whole
        epoch of feed forward steps and collects the results.
        """
        assert tf.train.checkpoint_exists(str(self.logdir)), \
            'No checkpoint found in logdir: {}'.format(self.logdir)

        results = []
        test_logs = str(self.logdir / 'test')
        handle_op = self.dataset.create_test_feed()
        kwargs = dict(checkpoint_dir=test_logs)
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
