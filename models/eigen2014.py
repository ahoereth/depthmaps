"""Implementation of Eigen et al 2014.

@inproceedings{Eigen2014,
  title = {Depth map prediction from a single image using a multi-scale deep network},
  author = {Eigen, David and Puhrsch, Christian and Fergus, Rob},
  booktitle = {Advances in neural information processing systems},
  pages = {2366--2374},
  year = {2014}
}
"""
import tensorflow as tf

from .lib import Model, Network


class Eigen2014(Model):
    input_shape = (304, 228, 1)  # Eigen et al 2014 uses grayscale input.
    target_shape = (74, 55, 1)
    batchsize = 32

    def build_network(self, inputs, targets, training=False):
        """Create a coarse/fine convolutional neural network.

        Note: inputs and targets are expected to be scaled from 0 to 1 when
        being passed to this network.
        """
        with tf.variable_scope('coarse'):
            coarse = tf.layers.conv2d(inputs, 96, 11, activation=tf.nn.relu,
                                      strides=4)
            coarse = tf.layers.max_pooling2d(coarse, pool_size=2, strides=2)
            coarse = tf.layers.conv2d(coarse, 256, 5, activation=tf.nn.relu,
                                      padding='same')
            coarse = tf.layers.max_pooling2d(coarse, pool_size=2, strides=2)
            coarse = tf.layers.conv2d(coarse, 384, 3, activation=tf.nn.relu,
                                      padding='same')
            coarse = tf.layers.conv2d(coarse, 384, 3, activation=tf.nn.relu,
                                      padding='same')
            coarse = tf.layers.conv2d(coarse, 256, 3, activation=tf.nn.relu,
                                      strides=2)
            coarse = tf.reshape(coarse, (-1, 8 * 6 * 256))
            coarse = tf.layers.dense(coarse, 4096, activation=tf.nn.relu)
            coarse = tf.layers.dropout(coarse, rate=.5, training=training)
            coarse = tf.layers.dense(coarse, 74 * 55)
            coarse = tf.reshape(coarse, (-1, 74, 55, 1))
        with tf.variable_scope('fine'):
            fine = tf.layers.conv2d(inputs, 63, 9, activation=tf.nn.relu,
                                    strides=2,)
            fine = tf.layers.max_pooling2d(fine, pool_size=2, strides=2)
            fine = tf.concat([fine, coarse], 3)  # Coarse results enter!!
            fine = tf.layers.conv2d(fine, 64, 5, activation=tf.nn.relu,
                                    padding='same')
            # Using sigmoid here which the paper does not because the targets
            # are scaled from 0 to 1 the same way the inputs are.
            outputs = tf.layers.conv2d(fine, 1, 5, padding='same',
                                       activation=tf.nn.sigmoid)
        with tf.variable_scope('loss'):
            tf.losses.mean_squared_error(targets, outputs)
            loss = tf.losses.get_total_loss()
            tf.summary.scalar('total', loss)
        with tf.variable_scope('optimizer'):
            train = tf.train.AdamOptimizer(1e-4).minimize(loss, self.step)
        return Network(outputs, train, loss)