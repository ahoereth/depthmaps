import tensorflow as tf


def lrelu(x, alpha=0.2):
    """Leaky rectifier."""
    return tf.maximum(alpha * x, x)


def soft_labels_like(like, value: bool):
    assert type(value) == bool
    if value:
        return tf.random_uniform(tf.shape(like), 0.7, 1.2, tf.float32)
    else:
        return tf.random_uniform(tf.shape(like), 0., 0.3, tf.float32)
