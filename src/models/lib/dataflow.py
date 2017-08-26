import tensorflow as tf
from tensorflow.contrib.data import Dataset


def read_image(filepath, image_size):
    read = tf.read_file(filepath)
    decoded = tf.image.decode_image(read)
    resized = tf.image.resize_images(decoded, out_shape)
    return resized


def to_float(images):
    """Convert uint8 images to float and scale them from -1 to 1."""
    return (tf.image.convert_image_dtype(images, tf.float32) - .5) * 2


class Dataflow:
    def __init__(self, filepath_tuples, shapes):
        self.input_shape, self.target_shape = shapes

        inputs, targets = zip(*filepath_tuples)

        self.dataset = Dataset.from_tensor_slices((inputs, targets))
        self.dataset = self.dataset.map(self._parse_files,
                                        num_threads=2,
                                        output_buffer_size=20)
        self.dataset = self.dataset.map(self._float_images)

    def get(self, batchsize=32):
        dataset = self.dataset.batch(batchsize)
        iterator = dataset.make_initializable_iterator()
        return iterator.get_next()

    def _parse_files(self, input_filepath, target_filepath):
        input_image = read_image(input_filepath, self.input_shape)
        target_image = read_image(target_filepath, self.target_shape)
        return input_image, target_image

    def _float_images(self, input_image, target_image):
        return to_float(input_image), to_float(target_image)
