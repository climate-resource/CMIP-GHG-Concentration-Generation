"""
Testing tools
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import iris
from numpy.typing import NDArray


def get_ndarrays_regression_array_contents(
    files_to_include: Iterable[Path],
    root_dir_output: Path,
    re_substitutions: tuple[tuple[str, str], ...] = (
        (
            r"v\d{8}",
            "vYYYYMMDD",
        ),
    ),
) -> dict[str, NDArray[Any]]:
    """
    Get content for `pytest-regression`'s ndarrays_regression fixture

    Parameters
    ----------
    files_to_include
        File(s) to include when creating the regression content

    root_dir_output
        Root directory of the path(s) in which the file(s) were written

    re_substitutions
        Regular expression substitutions to apply to the filenames

    Returns
    -------
    :
        Content which can be used with `ndarrays_regression`.
    """
    out = {}
    for input4mips_file in files_to_include:
        cubes = iris.load(input4mips_file)
        if len(cubes) > 1:
            raise NotImplementedError

        cube = cubes[0]
        key = cube.name()
        filepath_write = str(input4mips_file.relative_to(root_dir_output))
        for regexp, sub_val in re_substitutions:
            filepath_write = re.sub(
                regexp,
                sub_val,
                filepath_write,
            )

        key_write = f"{filepath_write}__{key}"
        out[key_write] = cube.data

    return out
