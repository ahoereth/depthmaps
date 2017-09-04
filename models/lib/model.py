"""Depth map generation base model."""
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from threading import Thread

import tensorflow as tf

from .feed_summary_saver_hook import FeedSummarySaverHook
from .utils import to_float


class Model:
    """Depth map generation base model."""
    input_shape = (0, 0, 0)
    target_shape = (0, 0, 0)
    batchsize = 32

    def __init__(self, dataset, checkpoint_dir=None):
        time = datetime.now().strftime('%y%m%d-%H%M')
        self.logdir = Path('logs') / type(self).__name__ / time

        # The checkpoint directory initially holds the location of a checkpoint
        # passed in from the outside and later on, if this model is being
        # trained directly, will be updated with its own checkpoint location.
        self.checkpoint_dir = checkpoint_dir

        self.dataset = dataset

        self.step = tf.train.get_or_create_global_step()

        self.training = tf.placeholder_with_default(False, None)

        # Resize and scale the test and train data, provide iterators.
        shapes = (self.input_shape, self.target_shape)
        feed, self.feedhandle = self.dataset.finalize(shapes, self.batchsize)
        self.inputs, self.targets = feed

        # Create the network.
        self.outputs, self.train_op = self.build_network(self.inputs,
                                                         self.targets,
                                                         self.training)

        tf.summary.image('inputs', self.inputs)
        tf.summary.image('targets', self.targets)
        tf.summary.image('outputs', self.outputs)
        self.summaries = tf.summary.merge_all()

    def build_network(self, inputs, targets, training=False):
        """Create the neural network."""
        raise NotImplementedError

    def train(self, epochs=1):
        handle_op = self.dataset.create_train_feed(epochs)
        test_handle_op = self.dataset.create_train_feed(epochs=-1)

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

        ckp = test_logs if self.checkpoint_dir is None else self.checkpoint_dir
        kwargs = dict(checkpoint_dir=ckp, hooks=hooks, config=config,
                      stop_grace_period_secs=10)

        # Model is being trained directly, update checkpoint location.
        self.checkpoint_dir = test_logs

        # Train the model.
        with tf.train.SingularMonitoredSession(**kwargs) as sess:
            handle = sess.raw_session().run(handle_op)
            while not sess.should_stop():
                sess.run(self.train_op, {self.feedhandle: handle,
                                         self.training: True})

    def evaluate(self):
        """Evaluate the model.

        Still passes the data through the model in the specified batchsize
        in order to prevent out of memory errors. Basically performs a whole
        epoch of feed forward steps and collects the results.
        """
        assert tf.train.checkpoint_exists(str(self.checkpoint_dir)), \
            'No checkpoint found in logdir: {}'.format(self.checkpoint_dir)

        results = []
        handle_op = self.dataset.create_test_feed()
        kwargs = dict(checkpoint_dir=self.checkpoint_dir)
        with tf.train.SingularMonitoredSession(**kwargs) as sess:
            handle = sess.raw_session().run(handle_op)
            while not sess.should_stop():
                inputs, targets, outputs = sess.run(
                    [self.inputs, self.targets, self.outputs],
                    {self.feedhandle: handle})
                results.extend(list(zip(inputs, outputs, targets)))
        return results
