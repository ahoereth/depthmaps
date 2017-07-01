import tensorflow as tf

from .lib import Model, Network


class Simple(Model):
    input_shape = (128, 96, 3)
    target_shape = (47, 31, 1)
    batchsize = 32

    def __init__(self, dataset):
        super(Simple, self).__init__(dataset=dataset)

    def build_network(self, inputs, targets):
        net = tf.layers.conv2d(inputs, 16, 16, 2, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 16, 8, 1, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 32, 4, 1, activation=tf.nn.relu)
        outputs = tf.layers.conv2d(net, 1, 1, activation=tf.identity)
        loss = tf.reduce_mean(tf.squared_difference(targets, outputs))
        train = tf.train.AdamOptimizer().minimize(loss)
        return Network(outputs, train, loss)
