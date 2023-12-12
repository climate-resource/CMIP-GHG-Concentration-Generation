"""
Test the workflow results

This is a regression test so the entire workflow is run
"""
from __future__ import annotations

import xarray as xr


def test_basic_workflow(basic_workflow_output_info, ndarrays_regression):
    """
    Test the basic workflow

    This workflow's runtime should be kept to a minimum so it can sensibly be
    used as a test. Less than 30 seconds to run the test is what we should be
    aiming for.
    """
    array_contents = {}
    for input4mips_file in (
        basic_workflow_output_info["root_dir_output"]
        / basic_workflow_output_info["run_id"]
        / "data"
        / "processed"
        / "input4MIPs"
    ).rglob("*.nc"):
        ds = xr.open_dataset(input4mips_file)
        for key, value in ds.data_vars.items():
            array_contents[
                f"{input4mips_file.relative_to(basic_workflow_output_info['root_dir_output'])}__{key}"
            ] = value.data

    ndarrays_regression.check(array_contents)
