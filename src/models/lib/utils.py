"""Utility functions."""
import tensorflow as tf

from tensorflow.python.client import device_lib


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


def to_tuple(*args):
    result = []
    for arg in args:
        try:
            result.extend(arg)
        except TypeError:
            result.extend((arg,))
    return tuple(result)


def get_available_gpus():
    """Get available GPUs.

    See https://stackoverflow.com/a/38580201
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.

    Returns:
        List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.

    See github.com/tensorflow/models/blob/7d238c5/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads
