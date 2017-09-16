import multiprocessing

import tensorflow as tf

import datasets
import models
from datasets import Dataviewer


FLAGS = tf.app.flags.FLAGS
CPUS = multiprocessing.cpu_count()


tf.app.flags.DEFINE_string('dataset', 'Make3D', 'Dataset to use. Defaults to '
                           'Make3D. One of: [Make3D, Make3D2, Nyu, Merged]')
tf.app.flags.DEFINE_string('model', 'Pix2Pix', 'Model to use. Defaults to '
                           'Pix2Pix. One of: [Simple, MultiScale, Pix2Pix, '
                           'Generator, Inference]')
tf.app.flags.DEFINE_string('checkpoint_dir', None, 'Directory containing '
                           'a checkpoint to load, has to fit the model.')
tf.app.flags.DEFINE_integer('epochs', 0, 'Number of epochs to train for. '
                            'Defaults to 0 which is needed when only running '
                            'inference using a pretrained model.')
tf.app.flags.DEFINE_integer('workers', CPUS, 'Number of threads to use. '
                            'Defaults to the count of available cores.')
tf.app.flags.DEFINE_boolean('cleanup_on_exit', False,
                            'Remove temporary files on exit.')
tf.app.flags.DEFINE_integer('test_split', 10, 'Percentage of samples to use '
                            'for evaluation during training. Defaults to 10. '
                            'Only relevant if  use_predefined_split is set to '
                            'False or when  there is no such predefined split '
                            'available.')
tf.app.flags.DEFINE_boolean('use_custom_test_split', False,
                            'Whether to not use the dataset\'s predefined '
                            'train/test split even if one is available. '
                            'Defaults to False.')


def main(argv=None):  # pylint: disable=unused-argument
    assert hasattr(datasets, FLAGS.dataset), 'No such dataset available.'
    assert hasattr(models, FLAGS.model), 'No such model available.'
    assert FLAGS.epochs > 0 or FLAGS.checkpoint_dir is not None, \
        'checkpoint_dir required when no training planned. Otherwise set ' \
        'a number of epochs to train for.'

    Dataset = getattr(datasets, FLAGS.dataset)
    Model = getattr(models, FLAGS.model)

    dataset = Dataset(cleanup_on_exit=FLAGS.cleanup_on_exit,
                      use_predefined_split=not FLAGS.use_custom_test_split,
                      test_split=FLAGS.test_split, workers=FLAGS.workers)
    model = Model(dataset, checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.epochs > 0:
        model.train(epochs=FLAGS.epochs)
    results = model.evaluate()

    if Dataviewer.AVAILABLE:
        keys = ('inputs', 'outputs')
        if dataset.has_targets:
            keys = keys + ('targets',)
        Dataviewer(results, name='Results', keys=keys,
                   cmaps={'outputs': 'gray', 'targets': 'gray'})


if __name__ == '__main__':
    tf.app.run()
