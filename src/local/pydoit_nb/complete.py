"""
Complete file generation
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path


def write_complete_file(complete_file: Path, contents: str | None = None) -> Path:
    """
    Write a complete file

    Parameters
    ----------
    complete_file
        Path in which to write the complete file

    contents
        Contents to write in the file. If ``None``, we simply write a timestamp
        into the file.

    Returns
    -------
        Path to the written file
    """
    if contents is None:
        contents = dt.datetime.now().strftime("%Y%m%d%H%M%S")

    with open(complete_file, "w") as fh:
        fh.write(contents)
