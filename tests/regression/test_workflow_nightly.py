"""
Test the workflow results

This is intended for use in the nightly CI, hence the run time is much longer.

This is a regression test so the entire workflow is run.
"""

from __future__ import annotations

import pytest

from local.testing import get_regression_values


@pytest.mark.nightly
def test_workflow_nightly(
    nightly_workflow_output_info, data_regression, ndarrays_regression
):
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

    metadata_check, array_check = get_regression_values(
        files_to_include=files_to_include,
        root_dir_output=nightly_workflow_output_info["root_dir_output"],
    )

    data_regression.check(metadata_check)

    ndarrays_regression.check(
        array_check,
        default_tolerance=dict(
            # TODO: dial this back down
            atol=1e-1,
            rtol=1,
        ),
    )
