import os
import sys
import tarfile
from pathlib import Path
from urllib.error import HTTPError
from urllib import request


def maybe_download(target: str, url: str, *, force=False):
    """Download a file if it does not exist at the specified target."""
    def progress(count, size, total):
        total = total / (1024 * 1024)
        size = (size / (1024 * 1024)) * count
        sys.stdout.write('\rProgress: {:.2f}/{:.2f}MB...'.format(size, total))

    target_path = Path(target)
    if target_path.suffix:
        target_dir = target_path.parent
    else:
        filename = os.path.basename(url)
        target_dir = target_path
        target_path = target_dir / filename

    didntexist = not target_path.exists()
    if didntexist or force:
        os.makedirs(str(target_dir), exist_ok=True)
        print('Fetching `{}`...'.format(target_path))
        try:
            request.urlretrieve(url, str(target_path), progress)
            print("")
        except HTTPError:
            print('Failed to download `{}`.'.format(url))
    else:
        print('`{}` already exists.'.format(target_path))

    return target_path, didntexist or force


def maybe_extract(filename, target_dir=None, targets=False, *, force=False):
    """Extract a file to the specified target directory."""
    file = Path(filename)
    if target_dir is None:
        target_dir = file.parent / file.name[:-len(''.join(file.suffixes))]

    didntexist = not target_dir.exists()
    if didntexist or force:
        os.makedirs(str(target_dir), exist_ok=True)
        with tarfile.open(str(file)) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, str(target_dir))
    else:
        print('`{}` already exists.'.format(target_dir))

    return target_dir, didntexist or force
