"""
Test the workflow results

This is intended for use in the nightly CI, hence the run time is much longer.

This is a regression test so the entire workflow is run.
"""

from __future__ import annotations

import pytest

from local.testing import get_ndarrays_regression_array_contents


@pytest.mark.nightly
def test_workflow_nightly(nightly_workflow_output_info, ndarrays_regression):
    """
    Test the nightly workflow
    """
    files_to_include = tuple(
        (
            nightly_workflow_output_info["root_dir_output"]
            / nightly_workflow_output_info["run_id"]
            / "data"
            / "processed"
            / "esgf-ready"
            / "input4MIPs"
        ).rglob("*.nc")
    )

    array_contents = get_ndarrays_regression_array_contents(
        files_to_include=files_to_include,
        root_dir_output=nightly_workflow_output_info["root_dir_output"],
    )

    ndarrays_regression.check(
        array_contents,
        default_tolerance=dict(
            # TODO: dial this back down
            atol=1e-1,
            rtol=1,
        ),
    )
