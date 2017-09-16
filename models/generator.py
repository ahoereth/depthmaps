"""Encode decoder image generator.

Based on Pix2Pix, without any fancy stuff.
"""
import tensorflow as tf

from .lib import Model, lrelu
from .pix2pix import Pix2Pix


class Generator(Pix2Pix):

        # print(get_available_gpus())
    def build_network(self, inputs, targets, training=False):
        global_step = tf.train.get_or_create_global_step()

        # Scale inputs and targets from -1 to 1.
        inputs = tf.subtract(inputs * 2, 1, name='scaled_inputs')
        targets = tf.subtract(targets * 2, 1, name='scaled_targets')
        tf.summary.histogram('input', inputs)
        tf.summary.histogram('target', targets)

        generator = self.make_generator(inputs)

        # Keep moving averages over the training and testing loss individually.
        trainema = tf.train.ExponentialMovingAverage(decay=0.999)
        testema = tf.train.ExponentialMovingAverage(decay=0.99)

        with tf.variable_scope('loss'):
            tf.losses.mean_squared_error(labels=targets, predictions=generator)
            loss = tf.losses.get_total_loss()
            tf.summary.scalar('live', loss)

        with tf.control_dependencies([trainema.apply([loss])]):
            optimizer = tf.train.AdamOptimizer(1e-4)
            train_op = optimizer.minimize(loss, global_step)

        # Scale outputs back to 0/1 range and add the test loss ema op.
        with tf.control_dependencies([testema.apply([loss])]):
            outputs = (generator + 1) / 2

        # Select the correct loss ema to summarize.
        loss_ema = tf.cond(training,
                           lambda: trainema.average(loss),
                           lambda: testema.average(loss))
        tf.summary.scalar('loss/ema', loss_ema)

        return outputs, train_op
