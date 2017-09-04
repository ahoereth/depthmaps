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

    def __init__(self, *args, cleanup_on_exit=True, **kwargs):
        self._mock_images(DATA_DIR / '..' / 'inference', self.directory)
        super().__init__(*args, **kwargs)

    def _mock_images(self, src, dst):
        """Creates input and mock depth image for each image in src."""
        mock = Image.fromarray(np.zeros(self.target_shape, dtype=np.uint8))
        for path in glob(str(src / '**/*.(jpg|png|gif)')):
            try:
                with Image.open(path) as img:
                    img = img.resize(self.input_shape)
            except (ValueError, OSError):
                print("Couldn't open {}".format(path))
            else:
                path = Path(path)
                filename = (path.parent / path.name)
                img.save(filename.with_suffix('.image.png'), 'PNG')
                mock.save(filename.with_suffix('.depth.png'), 'PNG')


if __name__ == '__main__':
    """Take a look at the data using `python -m datasets.inference`."""
    Inference().view()
