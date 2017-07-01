from collections import namedtuple
from queue import Queue, Empty
from threading import Thread

import tensorflow as tf

from .dataflow import Dataflow


Network = namedtuple('Network', ('output', 'train', 'loss'))


def to_float(images):
    return tf.image.convert_image_dtype(images, tf.float32)


class Model:
    input_shape = (0, 0, 0)
    target_shape = (0, 0, 0)
    batchsize = 32

    def __init__(self, dataset, workers=1):
        self.dataset = dataset
        self.session = tf.Session()

        self.training = tf.placeholder_with_default(False, None)

        # Create two dataflows, one for the train and one for test split.
        shapes = (self.input_shape, self.target_shape)
        self.train_dataflow = Dataflow(self.session, self.dataset.train_files,
                                       shapes, batchsize=self.batchsize,
                                       workers=workers)
        self.test_dataflow = Dataflow(self.session, self.dataset.test_files,
                                      shapes, len(self.dataset.test_files),
                                      workers=1)

        # Convert all images to float values from 0 to 1.
        train_inputs, train_targets = map(to_float, self.train_dataflow.out)
        test_inputs, test_targets = map(to_float, self.test_dataflow.out)

        # To handle the train and test split there are two networks which
        # share variables. This allows us to use them independently.
        build_network = tf.make_template('network', self.build_network,
                                         training=self.training)
        self.train_net = build_network(train_inputs, train_targets)
        self.test_net = build_network(test_inputs, test_targets)

        self.session.run(tf.global_variables_initializer())

        self.train_queue = Queue(len(self.dataset.train_files))
        self.workers = [Thread(target=self.train_worker, daemon=True)
                        for _ in range(workers)]

    def build_network(self, inputs, targets):
        """Create the neural network."""
        raise NotImplementedError

    def train_worker(self):
        while True:
            try:
                epoch, task = self.train_queue.get(timeout=5)
            except Empty:
                break

            if task == 'eval':
                loss = self.session.run(self.test_net.loss)
                print('Epoch {} with test loss {:.5f}.'.format(epoch, loss))
                continue

            self.session.run(self.train_net.train, {self.training: True})

    def train(self, epochs=1, test_frequency=1):
        """Train the model."""
        # Start workers.
        for worker in self.workers:
            worker.start()

        # Allow a specific amount of training steps with regular evaluations.
        for epoch in range(1, epochs + 1):
            for _ in range(len(self.dataset.train_files) // self.batchsize):
                self.train_queue.put((epoch, 'train'))
            if not epoch % test_frequency:
                self.train_queue.put((epoch, 'eval'))

        # Wait for all workers to be done.
        for worker in self.workers:
            worker.join()

    def evaluate(self):
        data, outputs = self.session.run([self.test_dataflow.out,
                                          self.test_net.output])
        inputs, targets = data
        return list(zip(inputs, targets, outputs))
