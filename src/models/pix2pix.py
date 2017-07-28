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

from .lib import Model, Network, lrelu, soft_labels_like


class Pix2Pix(Model):
    LAMBDA = 100

    input_shape = (256, 256, 3)
    target_shape = (256, 256, 1)
    batchsize = 1

    def __init__(self, *args, **kwargs):
        super(Pix2Pix, self).__init__(*args, **kwargs)

    @staticmethod
    def conv2d(inputs, num_outputs, kernel_size=(4, 4), strides=2,
               padding='SAME', activation=lrelu, norm=False, training=False):
        """Wrapper for tf.layers.conv2d with default parameters."""
        net = tf.layers.conv2d(inputs, num_outputs, kernel_size, strides,
                               padding=padding)
        if norm:
            net = tf.layers.batch_normalization(net, training=training)
        return activation(net)

    @staticmethod
    def conv2d_transpose(inputs, num_outputs, kernel_size=(4, 4), strides=2,
                         padding='SAME', activation=tf.nn.relu, norm=None):
        """Wrapper for tf.layers.conv2d_transpose with default parameters."""
        net = tf.layers.conv2d_transpose(inputs, num_outputs, kernel_size,
                                         strides, padding=padding,
                                         activation=activation)
        if norm is not None:
            net = tf.layers.batch_normalization(net, training=norm)
        return net  # activation(net)

    @classmethod
    def make_discriminator(cls, images, training):
        """Discriminator.

        Args:
            images: Stack of original and target images, either from generator
                or ground truths. Shape (BATCH, 256, 256, (input_c + output_c))
            training : True when in training, False when in testing step.

        Returns:
            (tf.Tensor): Sigmoid network output, single scalar in [0, 1].
            (tf.Tensor): Linear network output, single scalar.
            (List[tf.Operation]): Batch normalization update operations.
        """
        with tf.variable_scope('discriminator') as scope:
            net = cls.conv2d(images, 64)
            net = cls.conv2d(net, 128, norm=training)
            net = cls.conv2d(net, 256, norm=training)
            net = cls.conv2d(net, 512, norm=training)
            net = cls.conv2d(net, 1, (1, 1))
            logits = tf.layers.dense(net, 1, tf.identity)
            # out = tf.nn.sigmoid(logits)
            theta = scope.trainable_variables()
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
        return logits, theta, ops  # NOTE: No output returned

    @classmethod
    def make_generator(cls, images, training):
        """Generator.

        Args:
            images: Input images, either from generator or ground truths.
                Shape (BATCH, 256, 256, input_c)
            training: True when in training, False when in testing step.

        Returns:
            (tf.Tensor): Tanh network output, single channel shaped as input.
            (List[tf.Operation]): Batch normalization update operations.
        """
        with tf.variable_scope('generator') as scope:
            with tf.variable_scope('encoder'):  # 256x256
                net = images
                enc0 = cls.conv2d(net, 64)  # 128x128
                enc1 = cls.conv2d(enc0, 128, norm=training)  # 64x64
                enc2 = cls.conv2d(enc1, 256, norm=training)  # 32x32
                enc3 = cls.conv2d(enc2, 512, norm=training)  # 16x16
                enc4 = cls.conv2d(enc3, 512, norm=training)  # 8x8
                enc5 = cls.conv2d(enc4, 512, norm=training)  # 4x4
                enc6 = cls.conv2d(enc5, 512, norm=training)  # 2x2x512
                enc7 = cls.conv2d(enc6, 512, norm=training)  # 1x1x512
            with tf.variable_scope('decoder'):
                dec = cls.conv2d_transpose(enc7, 512, norm=training)  # 2x2
                dec = tf.layers.dropout(dec, .5)  # 512
                dec = tf.concat([dec, enc6], axis=-1)  # 1024
                dec = cls.conv2d_transpose(dec, 512, norm=training)  # 4x4
                dec = tf.layers.dropout(dec, .5)
                dec = tf.concat([dec, enc5], axis=-1)  # 1024
                dec = cls.conv2d_transpose(dec, 512, norm=training)  # 8x8
                dec = tf.layers.dropout(dec, .5)
                dec = tf.concat([dec, enc4], axis=-1)  # 1024
                dec = cls.conv2d_transpose(dec, 512, norm=training)  # 16x16
                dec = tf.concat([dec, enc3], axis=-1)  # 1024
                # dec = conv2d_transpose(dec, 512, norm=True,
                #                        training=training)
                dec = cls.conv2d_transpose(dec, 256, norm=training)  # 32x32
                dec = tf.concat([dec, enc2], axis=-1)  # 512
                dec = cls.conv2d_transpose(dec, 128, norm=training)  # 64x64
                dec = tf.concat([dec, enc1], axis=-1)  # 256
                dec = cls.conv2d_transpose(dec, 64, norm=training)  # 128x128
                dec = tf.concat([dec, enc0], axis=-1)  # 128
                out = cls.conv2d_transpose(dec, 1,
                                           activation=tf.nn.tanh)  # 256x256
                theta = scope.trainable_variables()
                ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            return out, theta, ops

    def build_network(self, inputs, targets, training=False):
        """Create a generative adversarial image generation network."""
        # Create generator -- also acts as the sample.
        generator, g_theta, g_ops = self.make_generator(inputs, training)

        # Create the two discriminator graphs, once with the ground truths
        # and once the generated depth maps as inputs.
        real = tf.concat([inputs, targets], axis=-1)
        fake = tf.concat([inputs, generator], axis=-1)
        remake_discriminator = tf.make_template('discriminator',
                                                self.make_discriminator,
                                                training=training)
        d_logits, d_theta, d_ops = remake_discriminator(real)
        d_logits_, _, _ = remake_discriminator(fake)

        # Keep moving averages of losses.
        d_ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        g_ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        def train_generator():
            labels = soft_labels_like(d_logits_, True)
            ganloss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_,
                                                              labels=labels)
            ganloss = tf.reduce_mean(ganloss)
            l1loss = tf.reduce_mean(tf.abs(targets - generator))
            loss = ganloss + self.LAMBDA * l1loss
            ema_op = g_ema.apply(loss)
            optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
            with tf.control_dependencies(g_ops + [ema_op]):
                train_op = optimizer.minimize(loss, var_list=g_theta,
                                              global_step=self.step)
            return train_op

        def train_discriminator():
            labels = soft_labels_like(d_logits, True)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
                                                           labels=labels)
            labels_ = soft_labels_like(d_logits_, False)
            loss_ = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_,
                                                            labels=labels_)
            loss = tf.reduce_mean(loss) + tf.reduce_mean(loss_)
            ema_op = d_ema.apply(loss)
            optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
            with tf.control_dependencies(d_ops + [ema_op]):
                train_op = optimizer.minimize(loss, var_list=d_theta,
                                              global_step=self.step)
            return train_op

        # Run train operations alternating.
        train = tf.cond(self.step % 2, train_generator, train_discriminator)
        return Network(generator, train, (d_ema, g_ema))
