"""
Test the workflow results

This is a regression test so the entire workflow is run
"""

from __future__ import annotations

import re

import iris


def test_basic_workflow(basic_workflow_output_info, ndarrays_regression):
    """
    Test the basic workflow

    This workflow's runtime should be kept to a minimum
    so it can sensibly be used as a test.
    Less than 30 seconds to run the test is what we should be aiming for.
    Obviously, this is not the case right now.
    """
    array_contents = {}
    for input4mips_file in (
        basic_workflow_output_info["root_dir_output"]
        / basic_workflow_output_info["run_id"]
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
                    basic_workflow_output_info["root_dir_output"]
                )
            ),
        )

        key_write = f"{filepath_write}__{key}"
        array_contents[key_write] = cube.data

    ndarrays_regression.check(
        array_contents, default_tolerance=dict(atol=1e-6, rtol=1e-3)
    )
