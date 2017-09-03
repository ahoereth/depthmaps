"""Dataset containing all other available depth image datasets."""

from .lib import DATA_DIR, Dataset
from .make3d import Make3D
from .make3d2 import Make3D2
from .nyu import Nyu


class Merged(Dataset):
    directory = DATA_DIR
    predefined_split_available = False

    def __init__(self, *args, **kwargs):
        # Prepare all the data and collect the temporary directories.
        self._tempdirs.append(Make3D()._tempdirs)
        self._tempdirs.append(Make3D2()._tempdirs)
        self._tempdirs.append(Nyu()._tempdirs)

        # This should merge all datasets because self.directory is the parent
        # diretory to all the data from the individual datasets.
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    """Take a look at the data using `python -m datasets.merged`."""
    Merged().view()
