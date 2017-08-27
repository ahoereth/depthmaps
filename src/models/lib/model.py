"""Depth map generation base model."""
from collections import namedtuple
from datetime import datetime
from pathlib import Path

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

        tf.train.create_global_step()
        self.step = tf.train.get_global_step()

        self.training = tf.placeholder_with_default(False, None)

        # Resize and scale the test and train data, provide iterators.
        testdata = dataset.test.map(self._preprocess).batch(self.batchsize)
        traindata = dataset.train.map(self._preprocess).batch(self.batchsize)
        self.test_iter = testdata.make_initializable_iterator()
        self.train_iter = traindata.make_initializable_iterator()
        self.test_inputs, self.test_targets = self.test_iter.get_next()
        train_inputs, train_targets = self.train_iter.get_next()

        # To handle the train and test split there are two networks which
        # share variables. This allows us to use them independently.
        self.train_net = self.build_network(train_inputs, train_targets,
                                            training=self.training)
        self.test_net = self.build_network(self.test_inputs, self.test_targets,
                                           training=self.training, reuse=True)

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

    def train(self, epochs=1, save_frequency=10):
        """Train the model.

        TODO: Implement logging summaries on test set.
        """
        counter = 0
        for epoch in range(1, epochs + 1):
            self.session.run(self.train_iter.initializer)
            while True:
                counter += 1
                try:
                    if not counter % 100:
                        step, log, _ = self.session.run([self.step,
                                                         self.summaries,
                                                         self.train_net.train],
                                                        {self.training: True})
                        self.writer.add_summary(log)
                    else:
                        step, _ = self.session.run([self.step,
                                                    self.train_net.train],
                                                   {self.training: True})
                except tf.errors.OutOfRangeError:
                    break  # epoch done
            if not save_frequency % epoch or epoch == epochs:
                self.saver.save(self.session, str(self.logdir), step)

    def evaluate(self):
        """Evaluate the model.

        Still passes the data through the model in the specified batchsize
        in order to prevent out of memory errors. Basically performs a whole
        epoch of feed forward steps and collects the results.
        """
        results = []
        self.session.run(self.test_iter.initializer)
        while True:
            try:
                inputs, targets, outputs = self.session.run(
                    [self.test_inputs, self.test_targets,
                     self.test_net.output])
            except tf.errors.OutOfRangeError:
                break
            results.extend(list(zip(inputs, targets, outputs)))
        return results

    def _preprocess(self, input_image, target_image):
        input_image = tf.image.resize_images(input_image, self.input_shape)
        target_image = tf.image.resize_images(target_image, self.target_shape)
        return to_float(input_image), to_float(target_image)
