"""Dataset for using custom images for inference."""
import os
from glob import glob
from pathlib import Path

from PIL import Image
import numpy as np

from .lib import DATA_DIR, Dataset


class Inference(Dataset):
    directory = DATA_DIR / 'inference'
    input_shape = (480, 320)
    target_shape = (120, 80)  # Used for mock target images.
    test_only = True
    has_targets = False

    def __init__(self, cleanup_on_exit=True, **kwargs):
        self._mock_images(DATA_DIR / '..' / 'inference', self.directory)
        super().__init__(cleanup_on_exit=cleanup_on_exit, **kwargs)

    def _mock_images(self, src, dst):
        """Creates input and mock depth image for each image in src."""
        dst = dst.resolve()
        os.makedirs(str(dst), exist_ok=True)
        mock = Image.fromarray(np.zeros(self.target_shape, dtype=np.uint8))
        for ext in ('jpg', 'png', 'gif'):
            for path in glob(str(src / '**/*.{}'.format(ext)), recursive=True):
                try:
                    with Image.open(path) as img:
                        img = img.resize(self.input_shape)
                except (ValueError, OSError):
                    print("Couldn't open {}".format(path))
                else:
                    path = Path(path)
                    filename = (dst / path.name)
                    img.save(filename.with_suffix('.image.png'), 'PNG')
                    mock.save(filename.with_suffix('.depth.png'), 'PNG')


if __name__ == '__main__':
    """Take a look at the data using `python -m datasets.inference`."""
    Inference().view()
