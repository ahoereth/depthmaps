from pathlib import Path

from .dataset import Dataset
from .utils import maybe_extract, maybe_download


DATA_DIR = Path.cwd() / 'tmp'
