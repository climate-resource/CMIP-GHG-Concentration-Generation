"""
Test the workflow results

This is intended for use in the CI,
hence we only test a very limited part of the workflow
to keep the runtime short.

This is a regression test so the entire workflow is run.
"""

from __future__ import annotations

from local.testing import get_regression_values


def test_workflow_basic(
    basic_workflow_output_info, data_regression, ndarrays_regression
):
    """
    Test the basic workflow
    """
    files_to_include = tuple(
        (
            basic_workflow_output_info["root_dir_output"]
            / basic_workflow_output_info["run_id"]
            / "data"
            / "processed"
            / "esgf-ready"
            / "input4MIPs"
        ).rglob("*.nc")
    )

    metadata_check, array_check = get_regression_values(
        files_to_include=files_to_include,
        root_dir_output=basic_workflow_output_info["root_dir_output"],
    )

    data_regression.check(metadata_check)

    ndarrays_regression.check(array_check, default_tolerance=dict(atol=1e-6, rtol=1e-3))
