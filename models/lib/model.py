"""Depth map generation base model."""
import os
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
        self.logdir = (Path('logs') / type(self).__name__ /
                       type(dataset).__name__ / time)

        # Save the list of test files.
        os.makedirs(str(self.logdir), exist_ok=True)
        with open(str(self.logdir / 'test_files.txt'), 'w') as file:
            file.write(str(dataset))

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
        outputs, train_op, losses = self.build_network(self.inputs,
                                                       self.targets,
                                                       self.training)

        # Summarize train and test loss.
        ema_train = tf.train.ExponentialMovingAverage(decay=0.99)
        ema_test = tf.train.ExponentialMovingAverage(decay=0.9)
        with tf.control_dependencies([ema_test.apply(losses)]):
            self.outputs = tf.identity(outputs)
        self.train_op = tf.group(train_op, ema_train.apply(losses))

        # Summarize losses depending on current phase.
        for i, loss in enumerate(losses):
            average = tf.cond(self.training,
                              lambda: ema_train.average(loss),
                              lambda: ema_test.average(loss))
            tf.summary.scalar('loss/{}'.format(i), average)

        # Summarize samples from the networks behavior.
        tf.summary.image('inputs', self.inputs, max_outputs=1)
        tf.summary.image('targets', self.targets, max_outputs=1)
        tf.summary.image('outputs', self.outputs, max_outputs=1)

        self.summaries = tf.summary.merge_all()

    def build_network(self, inputs, targets, training=False):
        """Create the neural network."""
        raise NotImplementedError

    def train(self, epochs=1):
        handle_op = self.dataset.create_train_feed(epochs)
        test_handle_op = self.dataset.create_test_feed(epochs=-1)

        test_logs = str(self.logdir / 'test')
        train_logs = str(self.logdir / 'train')

        # Keeping all checkpoints in order to actually use the best one for
        # inference in the future.
        saver = tf.train.Saver(max_to_keep=0)
        checker = tf.train.CheckpointSaverHook(checkpoint_dir=test_logs,
                                               save_secs=60 * 10,  # 10 minutes
                                               saver=saver)

        # Writing summaries individually for testing and training data.
        summarizer = tf.train.SummarySaverHook(output_dir=train_logs,
                                               summary_op=self.summaries,
                                               save_steps=200)
        tester = FeedSummarySaverHook({self.feedhandle: test_handle_op},
                                      output_dir=test_logs,
                                      summary_op=self.summaries,
                                      save_steps=200)

        # Log how many steps the model makes per second.
        timer = tf.train.StepCounterHook(output_dir=train_logs,
                                         every_n_steps=200)

        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = \
            tf.OptimizerOptions.ON_2

        hooks = [checker, summarizer, timer, tester]
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
