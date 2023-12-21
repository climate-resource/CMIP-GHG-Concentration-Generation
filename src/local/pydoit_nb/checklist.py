"""
Checklist file generation
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from doit.dependency import get_file_md5

CHECKLIST_FNAME: str = "checklist.chk"
"""Name used for checklist files"""


def get_checklist_file(directory: Path) -> Path:
    """
    Get the full file path for a checklist

    Parameters
    ----------
    directory
        Directory for which we want to get the checklist file

    Returns
    -------
        Path of the checklist file

    """
    return directory / CHECKLIST_FNAME


def is_checklist_file(fp: Path) -> bool:
    """
    Check if a file is a checklist file

    Parameters
    ----------
    fp
        The file to check

    Returns
    -------
        ``True`` if the file is a checklist file, otherwise ``False``
    """
    return fp.name == CHECKLIST_FNAME


def create_md5_dict(
    files: Iterable[Path],
    exclusions: Iterable[Callable[[Path], bool]] | None = None,
) -> dict[Path, str]:
    """
    Create dictionary of MD5 hashes for files

    Parameters
    ----------
    files
        Files to create hashes for

    exclusions
        An iterable of callables. These are applied to each file. If any of the
        results is ``True`` then the file is skipped and will not be included
        in the dictionary of calculated hashes.

    Returns
    -------
        Dictionary of MD5 hashes for each file which was not excluded. The keys
        are the file paths (as they appear in ``files``) and the values are
        the calculated MD5 hashes.
    """
    out = {}
    for fp in files:
        if exclusions is not None and any(excl(fp) for excl in exclusions):
            # Don't include this file
            continue

        out[fp] = get_file_md5(fp)

    return out


def generate_directory_checklist(
    directory: Path,
    checklist_file: Path | None = None,
    exclusions: Iterable[Callable[[Path], bool]] | None = None,
) -> Path:
    """
    Create a checklist file with checksums for all files in a directory

    Running this command multiple times should result in the same result. This
    enables the checklist to be used as a target for doit tasks.

    The resulting checklist file can also be used to verify the contents
    of a folder using the program `mdfsum` so can be included in any
    distributed results.

    .. code:: bash

        md5sum -c checklist.chk

    Parameters
    ----------
    directory
        Directory containing arbitary files (we haven't tested this on any file
        but any directory containing hashable data files is the intended target)

    checklist_file
        Where to write the checklist file. If not supplied, the result of
        `get_checklist_file(directory)` is used.

    exclusions
        Functions used to check if a file should be excluded or not. If not
        supplied, we use `[is_checklist_file]`.

    Returns
    -------
        Path of the generated checklist file

    Raises
    ------
    NotADirectoryError
        If ``directory`` doesn't exist or isn't a directory
    """
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    if checklist_file is None:
        checklist_file = get_checklist_file(directory)

    if exclusions is None:
        # By default, don't include self
        exclusions = [is_checklist_file]

    # sort to ensure same result for same set of files
    files = sorted([f for f in directory.rglob("*") if f.is_file()])

    md5s = create_md5_dict(files, exclusions=exclusions)

    with open(checklist_file, "w") as fh:
        for fp, md5 in md5s.items():
            fh.write(f"{md5}  {fp.relative_to(directory)}\n")

    return checklist_file
