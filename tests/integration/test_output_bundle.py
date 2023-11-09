"""
Tests of the output bundle
"""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest


@pytest.mark.coverage_breaker
def test_output_bundle_runs(basic_workflow_output_info, tmpdir):
    copied_output_dir = tmpdir / "copied-workflow-output"
    shutil.copytree(basic_workflow_output_info["root_output_dir"], copied_output_dir)

    assumed_raw_config_file = "dev-config-raw.yaml"
    run_id = "bundle-run"

    env_here = copy.deepcopy(os.environ)
    env_here.pop("VIRTUAL_ENV")

    # Handy in case you need to debug
    subprocess.run(
        (  # noqa: S603 # inputs come from us
            "poetry",
            "config",
            "virtualenvs.in-project",
            "true",
        ),
        cwd=copied_output_dir / basic_workflow_output_info["run_id"],
        env=env_here,
    )

    subprocess.run(
        (  # noqa: S603 # inputs come from us
            "poetry",
            "install",
            "--only",
            "main",
        ),
        cwd=copied_output_dir / basic_workflow_output_info["run_id"],
        env=env_here,
    )

    venv_check = subprocess.run(
        (  # noqa: S603 # inputs come from us
            "poetry",
            "run",
            "which",
            "doit",
        ),
        cwd=copied_output_dir / basic_workflow_output_info["run_id"],
        env=env_here,
        stdout=subprocess.PIPE,
    )
    assert str(copied_output_dir) in venv_check.stdout.decode(), "venv incorrectly set"

    # Check that the instructions in the README are correct too and use them for
    # the run
    with open(
        copied_output_dir / basic_workflow_output_info["run_id"] / "README.md"
    ) as fh:
        readme_contents = fh.read()

    expected_readme_line = (
        f"DOIT_CONFIGURATION_FILE={assumed_raw_config_file} "
        "poetry run doit run --verbosity=2"
    )
    assert expected_readme_line in readme_contents

    subprocess.run(
        expected_readme_line.split(" ")[1:],  # noqa: S603 # inputs come from us
        cwd=copied_output_dir / basic_workflow_output_info["run_id"],
        env={
            "DOIT_CONFIGURATION_FILE": assumed_raw_config_file,
            "DOIT_RUN_ID": run_id,
            **os.environ,
        },
    )

    for compare_file in [Path("data") / "processed" / "910_draw-table.csv"]:
        bundle_res = pd.read_csv(
            copied_output_dir
            / basic_workflow_output_info["run_id"]
            / "output-bundles"
            / "bundle-run"
            / compare_file
        )

        test_res = pd.read_csv(
            basic_workflow_output_info["root_output_dir"]
            / basic_workflow_output_info["run_id"]
            / compare_file
        )

        pdt.assert_frame_equal(bundle_res, test_res)
