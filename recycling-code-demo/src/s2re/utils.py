from pathlib import Path
from typing import Union


def get_file_size(path: Union[str, Path]) -> int:
    """Returns the size of a file in bytes"""
    return Path(path).stat().st_size


def get_dir_size(path: Union[str, Path]) -> int:
    """Returns the size of a directory in bytes"""
    root_directory = Path(path)
    return sum(
        f.stat().st_size for f in root_directory.glob("**/*") if f.is_file()
    )


def get_size(path: Union[str, Path]) -> int:
    """Returns the size of a file or directory in bytes"""
    path = Path(path)
    if path.is_dir():
        return get_dir_size(path)
    else:
        return get_file_size(path)
