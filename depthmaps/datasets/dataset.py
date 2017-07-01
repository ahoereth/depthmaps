import atexit
import os
import shutil


datadirectory = os.path.join(os.path.dirname(__file__), 'tmp')


class Dataset:
    input_shape = (0, 0)
    target_shape = (0, 0)

    def __init__(self, cleanup_on_exit=False):
        self.files = {}
        self.train_files = []
        self.test_files = []
        self._tempdirs = []
        self.feeder = None
        if cleanup_on_exit:
            atexit.register(self._cleanup)

    def __len__(self):
        return len(self.files['test_data'])

    def _cleanup(self):
        """Delete temporary folders on exit."""
        for path in self._tempdirs:
            shutil.rmtree(path, ignore_errors=True)
