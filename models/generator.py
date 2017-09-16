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

        # Create the same generator network structure as used by Isola 2016.
        generator = self.make_generator(inputs)

        tf.losses.mean_squared_error(labels=targets, predictions=generator)
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(tf.losses.get_total_loss(), global_step)

        # Scale outputs back to between 0 and 1
        outputs = tf.divide(generator + 1, 2, name='outputs')

        return outputs, train_op, [tf.losses.get_total_loss()]
