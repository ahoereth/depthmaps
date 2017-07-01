from collections import namedtuple

import tensorflow as tf

from .dataflow import Dataflow


Network = namedtuple('Network', ('output', 'train', 'loss'))


def to_float(images):
    return tf.image.convert_image_dtype(images, tf.float32)


class Model:
    input_shape = (0, 0, 0)
    target_shape = (0, 0, 0)
    batchsize = 32

    def __init__(self, dataset):
        self.dataset = dataset
        self.session = tf.Session()

        self.training = tf.placeholder_with_default(False, None)

        # Create two dataflows, one for the train and one for test split.
        shapes = (self.input_shape, self.target_shape)
        self.train_dataflow = Dataflow(self.session, self.dataset.train_files,
                                       shapes, batchsize=self.batchsize,
                                       workers=2)
        self.test_dataflow = Dataflow(self.session, self.dataset.test_files,
                                      shapes, len(self.dataset.test_files),
                                      workers=1)

        # Convert all images to float values from 0 to 1.
        train_inputs, train_targets = map(to_float, self.train_dataflow.out)
        test_inputs, test_targets = map(to_float, self.train_dataflow.out)

        # To handle the train and test split there are two networks which
        # share variables. This allows us to use them independently.
        build_network = tf.make_template('network', self.build_network,
                                         training=self.training)
        self.train_net = build_network(train_inputs, train_targets)
        self.test_net = build_network(test_inputs, test_targets)

        self.session.run(tf.global_variables_initializer())

    def build_network(self, inputs, targets):
        """Create the neural network."""
        raise NotImplementedError

    def train(self, epochs=1, test_frequency=1):
        """Train the model."""
        for epoch in range(1, epochs + 1):
            for _ in range(len(self.dataset.train_files) // self.batchsize):
                self.session.run(self.train_net.train, {self.training: True})
            if not epoch % test_frequency:
                loss = self.session.run(self.test_net.loss)
                print('Epoch {} with test loss {:.5f}.'.format(epoch, loss))
