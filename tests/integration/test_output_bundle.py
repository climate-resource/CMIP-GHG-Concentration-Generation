"""
Tests of the output bundle
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess

import pytest
import xarray as xr
import xarray.testing as xrt


@pytest.mark.coverage_breaker
def test_output_bundle_runs(basic_workflow_output_info, tmpdir):
    copied_output_dir = tmpdir / "copied-workflow-output"
    shutil.copytree(basic_workflow_output_info["root_dir_output"], copied_output_dir)

    assumed_raw_config_file = "ci-config-raw.yaml"
    run_id = "bundle-run"

    env_here = copy.deepcopy(os.environ)
    env_here.pop("CONDA_PREFIX")
    env_here.pop("PIXI_PROJECT_MANIFEST")

    subprocess.check_output(
        (  # noqa: S603 # inputs come from us
            "pixi",
            "install",
            "-e",
            "default",
        ),
        cwd=copied_output_dir / basic_workflow_output_info["run_id"],
        env=env_here,
    )

    venv_check = subprocess.check_output(
        (  # noqa: S603 # inputs come from us
            "pixi",
            "run",
            "which",
            "doit",
        ),
        cwd=copied_output_dir / basic_workflow_output_info["run_id"],
        env=env_here,
    )
    assert str(copied_output_dir) in venv_check.decode(), "venv incorrectly set"

    # Check that the instructions in the README are correct too and use them for
    # the run
    with open(
        copied_output_dir / basic_workflow_output_info["run_id"] / "README.md"
    ) as fh:
        readme_contents = fh.read()

    expected_readme_line = (
        f"DOIT_CONFIGURATION_FILE={assumed_raw_config_file} "
        "pixi run doit run --verbosity=2"
    )
    assert expected_readme_line in readme_contents

    command_to_run_base = expected_readme_line.split(" ")[1:]
    subprocess.check_output(
        [*command_to_run_base, "-n", "2"],  # noqa: S603 # inputs come from us
        cwd=copied_output_dir / basic_workflow_output_info["run_id"],
        env={
            "DOIT_CONFIGURATION_FILE": assumed_raw_config_file,
            "DOIT_RUN_ID": run_id,
            **env_here,
        },
        stderr=subprocess.PIPE,
    )

    for compare_file in (
        basic_workflow_output_info["root_dir_output"]
        / basic_workflow_output_info["run_id"]
        / "data"
        / "processed"
        / "input4MIPs"
    ).rglob("*.nc"):
        bundle_res = xr.open_dataset(
            copied_output_dir
            / compare_file.relative_to(basic_workflow_output_info["root_dir_output"])
        )

        test_res = xr.open_dataset(compare_file)

        xrt.assert_equal(test_res, bundle_res)
