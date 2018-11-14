"""Implementation of Isola et al 2014.

@article{Isola2016,
  title = {Image-to-image translation with conditional adversarial networks},
  author = {Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal = {arXiv preprint arXiv:1611.07004},
  year = {2016}
}
"""
import tensorflow as tf

from .lib import Model, lrelu


class Pix2Pix(Model):
    LAMBDA = 100

    input_shape = (256, 256, 3)
    target_shape = (256, 256, 1)
    batchsize = 1

    @staticmethod
    def conv2d(inputs,
               num_outputs,
               kernel_size=(4, 4),
               strides=2,
               padding='SAME',
               activation=lrelu,
               norm=False):
        """Wrapper for tf.layers.conv2d with default parameters.

        Note the following quote, which also applies to conv2d_transpose:
        > We apply batch normalization using the statistics of
        > the test batch, rather than aggregated statistics of the training
        > batch. This approach to batch normalization, when the
        > batch size is set to 1, has been termed “instance normalization"
        > and has been demonstrated to be effective at image generation tasks.
        """
        init = tf.random_normal_initializer(0, 0.02)
        net = tf.layers.conv2d(
            inputs,
            num_outputs,
            kernel_size,
            strides,
            padding=padding,
            kernel_initializer=init)
        if norm:
            net = tf.layers.batch_normalization(net, training=True)
        return activation(net)

    @staticmethod
    def conv2d_transpose(inputs,
                         num_outputs,
                         kernel_size=(4, 4),
                         strides=2,
                         padding='SAME',
                         activation=tf.nn.relu,
                         norm=None):
        """Wrapper for tf.layers.conv2d_transpose with default parameters."""
        init = tf.random_normal_initializer(0, 0.02)
        net = tf.layers.conv2d_transpose(
            inputs,
            num_outputs,
            kernel_size,
            strides,
            padding=padding,
            kernel_initializer=init)
        if norm is not None:
            net = tf.layers.batch_normalization(net, training=True)
        return activation(net)

    @classmethod
    def make_discriminator(cls, images):
        """Discriminator.

        Args:
            images (tf.Tensor): Stack of original and target images, either
                from generator or ground truths.
                Shape (BATCH, 256, 256, (input_c + output_c))
            training (tf.Tensor): True when in training,
                False when in testing step.

        Returns:
            (tf.Tensor): Sigmoid network output, single scalar in [0, 1].
        """
        net = cls.conv2d(images, 64)  # 128x128
        net = cls.conv2d(net, 128, norm=True)  # 64x64
        net = cls.conv2d(net, 256, norm=True)  # 32x32
        net = cls.conv2d(net, 512, (1, 1), norm=True)  # 31x31
        net = cls.conv2d(net, 1, (1, 1), activation=tf.nn.sigmoid)
        return net  # 30x30

    @classmethod
    def make_generator(cls, images):
        """Generator.

        Args:
            images (tf.Tensor) : Input images, either from generator or ground
                truths. Shape (BATCH, 256, 256, input_c)
            training (tf.Tensor): True when in training,
                                  False when in testing step.

        Returns:
            (tf.Tensor): Tanh network output, single channel shaped as input.
        """
        with tf.variable_scope('encoder'):  # 256x256
            net = images
            enc0 = cls.conv2d(net, 64)  # 128x128
            enc1 = cls.conv2d(enc0, 128, norm=True)  # 64x64
            enc2 = cls.conv2d(enc1, 256, norm=True)  # 32x32
            enc3 = cls.conv2d(enc2, 512, norm=True)  # 16x16
            enc4 = cls.conv2d(enc3, 512, norm=True)  # 8x8
            enc5 = cls.conv2d(enc4, 512, norm=True)  # 4x4
            enc6 = cls.conv2d(enc5, 512, norm=True)  # 2x2x512
            enc7 = cls.conv2d(enc6, 512, norm=True)  # 1x1x512
        tf.summary.histogram('encoded', enc7)
        with tf.variable_scope('decoder'):
            dec = cls.conv2d_transpose(enc7, 512, norm=True)  # 2x2
            dec = tf.layers.dropout(dec, .5)  # 512
            dec = tf.concat([dec, enc6], axis=-1)  # 1024
            dec = cls.conv2d_transpose(dec, 512, norm=True)  # 4x4
            dec = tf.layers.dropout(dec, .5)
            dec = tf.concat([dec, enc5], axis=-1)  # 1024
            dec = cls.conv2d_transpose(dec, 512, norm=True)  # 8x8
            dec = tf.layers.dropout(dec, .5)
            dec = tf.concat([dec, enc4], axis=-1)  # 1024
            dec = cls.conv2d_transpose(dec, 512, norm=True)  # 16x16
            dec = tf.concat([dec, enc3], axis=-1)  # 1024
            dec = cls.conv2d_transpose(dec, 256, norm=True)  # 32x32
            dec = tf.concat([dec, enc2], axis=-1)  # 512
            dec = cls.conv2d_transpose(dec, 128, norm=True)  # 64x64
            dec = tf.concat([dec, enc1], axis=-1)  # 256
            dec = cls.conv2d_transpose(dec, 64, norm=True)  # 128x128
            dec = tf.concat([dec, enc0], axis=-1)  # 128
            out = cls.conv2d_transpose(dec, 1, activation=tf.nn.tanh)
        tf.summary.histogram('output', out)
        return out  # 256x256

        # print(get_available_gpus())
    def build_network(self, inputs, targets, training=False):
        """Create a generative adversarial image generation network.

        Note: inputs and targets are expected to be scaled from 0 to 1 when
        being passed to this network. This method handles scaling them to
        the tanh range and back.
        """
        global_step = tf.train.get_or_create_global_step()

        # Scale inputs and targets from -1 to 1.
        inputs = tf.subtract(inputs * 2, 1., name='scaled_inputs')
        targets = tf.subtract(targets * 2, 1., name='scaled_targets')
        tf.summary.histogram('input', inputs)
        tf.summary.histogram('target', targets)

        # Create generator.
        with tf.variable_scope('generator/net') as g_net:
            generator = self.make_generator(inputs)

        # Create the two discriminator graphs, once with the ground truths
        # and once the generated depth maps as inputs.
        with tf.variable_scope('discriminator') as d_net:  # Real
            real = tf.concat([inputs, targets], axis=-1, name='input/real')
            d_real = self.make_discriminator(real)
        with tf.variable_scope('discriminator', reuse=True):  # Fake
            fake = tf.concat([inputs, generator], axis=-1, name='input/fake')
            d_fake = self.make_discriminator(fake)

        with tf.variable_scope('generator/loss'):
            g_loss_gan = tf.reduce_mean(-tf.log(d_fake + 1e-12), name='gan')
            g_loss_l1 = tf.reduce_mean(tf.abs(targets - generator), name='l1')
            g_loss = g_loss_gan + self.LAMBDA * g_loss_l1

        with tf.variable_scope('discriminator/loss'):
            d_loss_real = tf.log(d_real + 1e-12, name='real')
            d_loss_fake = tf.log(1 - d_fake + 1e-12, name='fake')
            d_loss = tf.reduce_mean(-(d_loss_real + d_loss_fake))

        def train_generator():
            with tf.variable_scope('generator/optimizer'):
                g_theta = g_net.trainable_variables()
                optimizer = tf.train.AdamOptimizer(1e-4)
                return optimizer.minimize(g_loss, global_step, g_theta)

        def train_discriminator():
            with tf.variable_scope('discriminator/optimizer'):
                d_theta = d_net.trainable_variables()
                optimizer = tf.train.AdamOptimizer(1e-4)
                return optimizer.minimize(d_loss, global_step, d_theta)

        # Run train operations alternating.
        train = tf.cond(
            tf.cast(global_step % 2, tf.bool), train_generator,
            train_discriminator)

        # Scale outputs back to 0/1 range.
        outputs = (generator + 1) / 2.

        return outputs, train, (g_loss, d_loss)
