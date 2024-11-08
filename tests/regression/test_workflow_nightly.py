"""
Test the workflow results

This is intended for use in the nightly CI, hence the run time is much longer.

This is a regression test so the entire workflow is run.
"""

from __future__ import annotations

import re

import iris
import pytest


@pytest.mark.nightly
def test_workflow_nightly(nightly_workflow_output_info, ndarrays_regression):
    """
    Test the nightly workflow
    """
    array_contents = {}
    for input4mips_file in (
        nightly_workflow_output_info["root_dir_output"]
        / nightly_workflow_output_info["run_id"]
        / "data"
        / "processed"
        / "esgf-ready"
        / "input4MIPs"
    ).rglob("*.nc"):
        cubes = iris.load(input4mips_file)
        if len(cubes) > 1:
            raise NotImplementedError

        cube = cubes[0]
        key = cube.name()
        filepath_write = re.sub(
            r"v\d{8}",
            "vYYYYMMDD",
            str(
                input4mips_file.relative_to(
                    nightly_workflow_output_info["root_dir_output"]
                )
            ),
        )

        key_write = f"{filepath_write}__{key}"
        array_contents[key_write] = cube.data

    ndarrays_regression.check(
        array_contents, default_tolerance=dict(atol=1e-6, rtol=1e-3)
    )
