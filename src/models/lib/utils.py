"""Utility functions."""
import tensorflow as tf


def lrelu(x, alpha=0.2, name='lrelu'):
    """Leaky rectifier."""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * tf.abs(x)


def soft_labels_like(like, value: bool):
    assert type(value) == bool
    if value:
        return tf.random_uniform(tf.shape(like), 0.7, 1.2, tf.float32)
    else:
        return tf.random_uniform(tf.shape(like), 0., 0.3, tf.float32)


def to_float(images):
    """Convert uint8 images to float and scale them from -1 to 1."""
    return (tf.image.convert_image_dtype(images, tf.float32) - .5) * 2
