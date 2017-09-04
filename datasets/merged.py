"""Dataset containing all other available depth image datasets."""

from .lib import DATA_DIR, Dataset
from .make3d import Make3D
from .make3d2 import Make3D2
from .nyu import Nyu


class Merged(Dataset):
    directory = DATA_DIR
    has_predefined_split = False

    def __init__(self, *args, **kwargs):
        # Make sure each dataset is available, prepare their data.
        Make3D()
        Make3D2()
        Nyu()

        # This should merge all datasets because self.directory is the parent
        # diretory to all the data from the individual datasets.
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    """Take a look at the data using `python -m datasets.merged`."""
    Merged().view()
