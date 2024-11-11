"""
Test the workflow results

This is intended for use in the CI,
hence we only test a very limited part of the workflow
to keep the runtime short.

This is a regression test so the entire workflow is run.
"""

from __future__ import annotations

from local.testing import get_ndarrays_regression_array_contents


def test_workflow_basic(basic_workflow_output_info, ndarrays_regression):
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

    array_contents = get_ndarrays_regression_array_contents(
        files_to_include=files_to_include,
        root_dir_output=basic_workflow_output_info["root_dir_output"],
    )

    ndarrays_regression.check(
        array_contents, default_tolerance=dict(atol=1e-6, rtol=1e-3)
    )
