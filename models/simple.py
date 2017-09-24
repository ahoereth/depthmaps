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
        inputs = tf.subtract(inputs * 2, 1., name='scaled_inputs')
        targets = tf.subtract(targets * 2, 1., name='scaled_targets')
        tf.summary.histogram('input', inputs)
        tf.summary.histogram('target', targets)

        with tf.variable_scope('network'):
            net = tf.layers.conv2d(inputs, 16, 16, 2, activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 32, 8, 1, activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 48, 4, 1, activation=tf.nn.relu)
            net = tf.layers.conv2d(net, 1, 1, activation=tf.nn.tanh)

        global_step = tf.train.get_or_create_global_step()
        loss = tf.losses.mean_squared_error(targets, net)
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss, global_step)

        # Scale outputs back to between 0 and 1
        outputs = tf.divide(net + 1., 2., name='outputs')
        return outputs, train_op, [loss]
