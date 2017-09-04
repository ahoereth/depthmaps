import multiprocessing

import tensorflow as tf

import datasets
import models
from datasets import Dataviewer


FLAGS = tf.app.flags.FLAGS
CPUS = multiprocessing.cpu_count()


tf.app.flags.DEFINE_string('dataset', 'Make3D', 'Dataset to use.')
tf.app.flags.DEFINE_string('model', 'Pix2Pix', 'Model to use.')
tf.app.flags.DEFINE_string('checkpoint_dir', None, 'Directory containing '
                           'a checkpoint to load, has to fit the model.')
tf.app.flags.DEFINE_integer('epochs', 0, 'Number of epochs to train for.')
tf.app.flags.DEFINE_integer('workers', CPUS, 'Number of threads to use. '
                            'Defaults to the count of available cpus.')
tf.app.flags.DEFINE_boolean('cleanup_on_exit', False,
                            'Remove temporary files on exit.')
tf.app.flags.DEFINE_integer('test_split', 10, 'Percentage of samples to be'
                            'used for evaluation during training.')
tf.app.flags.DEFINE_boolean('use_predefined_split', False,
                            'Whether to use  the dataset\'s predefined '
                            'train/test split if one is available.')


def main(argv=None):  # pylint: disable=unused-argument
    assert hasattr(datasets, FLAGS.dataset), 'No such dataset available.'
    assert hasattr(models, FLAGS.model), 'No such model available.'

    Dataset = getattr(datasets, FLAGS.dataset)
    Model = getattr(models, FLAGS.model)

    dataset = Dataset(workers=FLAGS.workers)
    model = Model(dataset, checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.epochs > 0:
        model.train(epochs=FLAGS.epochs)
    results = model.evaluate()
    print(len(results))

    if Dataviewer.AVAILABLE:
        Dataviewer(results, name='Results', keys=['image', 'result', 'depth'],
                   cmaps={'depth': 'gray', 'result': 'gray'})


if __name__ == '__main__':
    tf.app.run()
