import tensorflow as tf

from .lib import Model


class Simple(Model):
    input_shape = (128, 96, 3)
    target_shape = (47, 31, 1)
    batchsize = 32

    def build_network(self, inputs, targets, training=False):
        """Build a simple fully convolutional model.

        Note: inputs and targets are expected to be scaled from 0 to 1 when
        being passed to this network.
        """
        with tf.variable_scope('network'):
            net = tf.layers.conv2d(inputs, 16, 16, 2, activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 16, 8, 1, activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 32, 4, 1, activation=tf.nn.relu)
            # Using sigmoid to scale output images from 0 to 1.
            outputs = tf.layers.conv2d(net, 1, 1, activation=tf.nn.sigmoid)

        global_step = tf.train.get_or_create_global_step()
        loss = tf.losses.mean_squared_error(targets, outputs)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step)

        return outputs, train_op, [loss]
