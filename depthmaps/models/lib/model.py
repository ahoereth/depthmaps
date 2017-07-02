from collections import namedtuple
from queue import Queue, Empty
from threading import Thread

import tensorflow as tf

from .dataflow import Dataflow


Network = namedtuple('Network', ('output', 'train', 'loss'))


def to_float(images):
    """Convert uint8 images to float and scale them from -1 to 1."""
    return (tf.image.convert_image_dtype(images, tf.float32) - .5) * 2


class Model:
    input_shape = (0, 0, 0)
    target_shape = (0, 0, 0)
    batchsize = 32

    def __init__(self, dataset, workers=1, checkpoint=None):
        self.dataset = dataset
        self.session = tf.Session()

        tf.train.create_global_step()
        self.step = tf.train.get_global_step()

        self.training = tf.placeholder_with_default(False, None)

        # Create two dataflows, one for the train and one for test split.
        shapes = (self.input_shape, self.target_shape)
        self.train_dataflow = Dataflow(self.session, self.dataset.train_files,
                                       shapes, batchsize=self.batchsize,
                                       workers=workers)
        self.test_dataflow = Dataflow(self.session, self.dataset.test_files,
                                      shapes, batchsize=self.batchsize,
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

        self.saver = tf.train.Saver(max_to_keep=1,
                                    keep_checkpoint_every_n_hours=1)
        if checkpoint is not None:
            self.saver.restore(checkpoint)
        else:
            self.session.run(tf.global_variables_initializer())

        self.train_queue = Queue(len(self.dataset.train_files))
        self.workers = [Thread(target=self.train_worker, daemon=True)
                        for _ in range(workers)]

    def build_network(self, inputs, targets):
        """Create the neural network."""
        raise NotImplementedError

    def train_worker(self):
        """Work through the train queue and perform the specified task."""
        while True:
            try:
                epoch, task = self.train_queue.get(timeout=2)
            except Empty:
                break

            if task == 'save':
                self.saver.save(self.session, type(self).__name__,
                                global_step=epoch)
            elif task == 'done':
                print('Epoch {} done.'.format(epoch))
            elif task == 'eval':
                loss = self.session.run(self.test_net.loss)
                print('Epoch {} with test loss {:.5f}.'.format(epoch, loss))
            elif task == 'verbose':
                _, loss = self.session.run(
                    [self.train_net.train, self.train_net.loss],
                    {self.training: True})
                print('Epoch {} with train loss {:.5f}.'.format(epoch, loss))
            else:
                self.session.run(self.train_net.train, {self.training: True})

    def train(self, epochs=1, test_frequency=1, test_on_testset=False,
              save_frequency=10):
        """Train the model."""
        # Start workers.
        for worker in self.workers:
            worker.start()

        # Allow a specific amount of training steps with regular evaluations.
        for epoch in range(1, epochs + 1):
            for _ in range(len(self.dataset.train_files) // self.batchsize):
                self.train_queue.put((epoch, 'train'))
            if test_frequency > 0 and not epoch % test_frequency:
                if test_on_testset:
                    self.train_queue.put((epoch, 'evaluate'))
                else:
                    self.train_queue.put((epoch, 'verbose'))
            elif test_frequency <= 0:
                self.train_queue.put((epoch, 'done'))
            if not epoch % save_frequency:
                self.train_queue.put((epoch, 'save'))

        # Wait for all workers to be done.
        for worker in self.workers:
            worker.join()

    def evaluate(self):
        """Evaluate the model.

        Still passes the data through the model in the specified batchsize
        in order to prevent out of memory errors. Basically performs a whole
        epoch of feed forward steps and collects the results.
        """
        inputs, targets, outputs = [], [], []
        for _ in range(len(self.dataset.test_files) // self.batchsize):
            data, output = self.session.run([self.test_dataflow.out,
                                             self.test_net.output])
            inputs.extend(data[0])
            targets.extend(data[1])
            outputs.extend(output)
        return list(zip(inputs, targets, outputs))
