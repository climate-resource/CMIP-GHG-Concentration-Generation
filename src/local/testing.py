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


def get_regression_values(
    files_to_include: Iterable[Path],
    root_dir_output: Path,
    metadata_keys_to_overwrite: tuple[str, ...] = ("creation_date", "tracking_id"),
    re_substitutions: tuple[tuple[str, str], ...] = (
        (
            r"v\d{8}",
            "vYYYYMMDD",
        ),
    ),
) -> tuple[dict[str, dict[str, str]], dict[str, NDArray[Any]]]:
    """
    Get value for `pytest-regression`-based tests

    Parameters
    ----------
    files_to_include
        File(s) to include when creating the regression content

    root_dir_output
        Root directory of the path(s) in which the file(s) were written

    metadata_keys_to_overwrite
        Metadata keys to overwrite with their names because the values are unstable.

    re_substitutions
        Regular expression substitutions to apply to the filenames

    Returns
    -------
    :
        Values which can be used with `data_regression` and `ndarrays_regression`.
    """
    out_data_regression = {}
    out_ndarrays_regression = {}
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

        metadata_to_check = {
            k: v if k not in metadata_keys_to_overwrite else k for k, v in cube.attributes.items()
        }
        out_data_regression[key_write] = metadata_to_check

        out_ndarrays_regression[key_write] = cube.data

    return out_data_regression, out_ndarrays_regression
