"""Implementation of Isola et al 2014.

@article{Isola2016,
  title = {Image-to-image translation with conditional adversarial networks},
  author = {Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal = {arXiv preprint arXiv:1611.07004},
  year = {2016}
}
"""
import tensorflow as tf

from .lib import Model, Network, lrelu, soft_labels_like


class Pix2Pix(Model):
    LAMBDA = 100

    input_shape = (256, 256, 3)
    target_shape = (256, 256, 1)
    batchsize = 1

    @staticmethod
    def conv2d(inputs, num_outputs, kernel_size=(4, 4), strides=2,
               padding='SAME', activation=lrelu, norm=None):
        """Wrapper for tf.layers.conv2d with default parameters."""
        init = tf.random_normal_initializer(0, 0.02)
        net = tf.layers.conv2d(inputs, num_outputs, kernel_size, strides,
                               padding=padding,
                               kernel_initializer=init)
        if norm is not None:
            net = tf.layers.batch_normalization(net, training=norm)
        return activation(net)

    @staticmethod
    def conv2d_transpose(inputs, num_outputs, kernel_size=(4, 4), strides=2,
                         padding='SAME', activation=tf.nn.relu, norm=None):
        """Wrapper for tf.layers.conv2d_transpose with default parameters."""
        init = tf.random_normal_initializer(0, 0.02)
        net = tf.layers.conv2d_transpose(inputs, num_outputs, kernel_size,
                                         strides, padding=padding,
                                         kernel_initializer=init)
        if norm is not None:
            net = tf.layers.batch_normalization(net, training=norm)
        return activation(net)

    @classmethod
    def make_discriminator(cls, images, training, reuse=None):
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
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            net = cls.conv2d(images, 64)  # 128x128
            net = cls.conv2d(net, 128, norm=training)  # 64x64
            net = cls.conv2d(net, 256, norm=training)  # 32x32
            net = cls.conv2d(net, 512, (1, 1), norm=training)  # 31x31
            net = cls.conv2d(net, 1, (1, 1), activation=tf.nn.sigmoid)  # 30x30
            theta = scope.trainable_variables()
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
        return net, theta, ops

    @classmethod
    def make_generator(cls, images, training):
        """Generator.

        Args:
            images: Input images, either from generator or ground truths.
                Shape (BATCH, 256, 256, input_c)
            training (tf.Tensor): True when in training,
                                  False when in testing step.

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
        inputs = inputs * 2 - 1  # scale from -1 to 1

        # Create generator.
        generator, g_theta, g_ops = self.make_generator(inputs, training)
        tf.contrib.layers.summarize_tensors(g_ops)

        # Create the two discriminator graphs, once with the ground truths
        # and once the generated depth maps as inputs.
        real = tf.concat([inputs, targets], axis=-1)
        fake = tf.concat([inputs, generator], axis=-1)
        d_real, d_theta, d_ops = self.make_discriminator(real, training)
        d_fake, _, _ = self.make_discriminator(fake, training, reuse=True)
        tf.contrib.layers.summarize_tensors(d_ops)

        # Keep moving averages over the training and testing loss individually.
        train_emas = tf.train.ExponentialMovingAverage(decay=0.999)
        test_emas = tf.train.ExponentialMovingAverage(decay=0.999)

        with tf.variable_scope('generator/loss'):
            g_loss_gan = tf.reduce_mean(-tf.log(d_fake + 1e-12))
            g_loss_l1 = tf.reduce_mean(tf.abs(targets - generator))
            g_loss = g_loss_gan + self.LAMBDA * g_loss_l1
            with tf.name_scope('summaries'):
                ema_g_train_op = train_emas.apply([g_loss])
                ema_g_test_op = test_emas.apply([g_loss])
                tf.summary.scalar('train', train_emas.average(g_loss))
                tf.summary.scalar('test', test_emas.average(g_loss))

        with tf.variable_scope('discriminator/loss'):
            d_loss_real = tf.log(d_real + 1e-12)
            d_loss_fake = tf.log(1 - d_fake + 1e-12)
            d_loss = tf.reduce_mean(-(d_loss_real + d_loss_fake))
            with tf.name_scope('summaries'):
                ema_d_train_op = train_emas.apply([d_loss])
                ema_d_test_op = test_emas.apply([d_loss])
                tf.summary.scalar('train', train_emas.average(d_loss))
                tf.summary.scalar('test', test_emas.average(d_loss))

        def train_generator():
            with tf.variable_scope('generator/optimizer'):
                with tf.control_dependencies(g_ops + [ema_g_train_op]):
                    optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
                    return optimizer.minimize(g_loss, self.step, g_theta)

        def train_discriminator():
            with tf.variable_scope('discriminator/optimizer'):
                with tf.control_dependencies(d_ops + [ema_d_train_op]):
                    optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
                    return optimizer.minimize(d_loss, self.step, d_theta)

        # Run train operations alternating.
        train = tf.cond(tf.cast(self.step % 2, tf.bool),
                        train_generator,
                        train_discriminator)

        # Scale generation output from 0 to 1.
        with tf.variable_scope('testing'):
            with tf.control_dependencies([ema_g_test_op, ema_d_test_op]):
                outputs = (generator + 1) / 2

        return Network(outputs, train, None)
