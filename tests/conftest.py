"""
Configuration file for pytest
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT_DIR = Path(__file__).parent.parent


def get_workflow_results(
    config_file: str,
    root_dir_output: str,
    run_id: str,
) -> dict[str, str | Path]:
    """
    Get the workflow results

    Parameters
    ----------
    config_file
        Config file to use in the run (relative to the root of the repository)

    root_dir_output
        Root directory in which to write the output (relative to the root of the repository)

    run_id
        Run ID to use for the output

    Returns
    -------
    :
        Workflow results
    """
    subprocess.run(
        (  # noqa: S603 # inputs come from us
            "doit",
            "run",
            "--verbosity=2",
            "-n",
            "2",
        ),
        cwd=REPO_ROOT_DIR,
        env={
            "DOIT_CONFIGURATION_FILE": config_file,
            "DOIT_ROOT_DIR_OUTPUT": root_dir_output,
            "DOIT_RUN_ID": run_id,
            **os.environ,
        },
        check=True,
    )

    return {
        "root_dir_output": REPO_ROOT_DIR / root_dir_output,
        "run_id": run_id,
    }


@pytest.fixture(scope="session")
def basic_workflow_output_info() -> dict[str, str | Path]:
    """
    Run the basic workflow and get the output info
    """
    config_file = "ci-config.yaml"
    root_dir_output = "output-bundles-tests"
    run_id = "test-basic-workflow"

    return get_workflow_results(
        config_file=config_file,
        root_dir_output=root_dir_output,
        run_id=run_id,
    )


@pytest.fixture(scope="session")
def nightly_workflow_output_info() -> dict[str, str | Path]:
    """
    Run the nightly workflow and get the output info
    """
    config_file = "ci-nightly-config.yaml"
    root_dir_output = "output-bundles-tests"
    run_id = "test-nightly-workflow"

    return get_workflow_results(
        config_file=config_file,
        root_dir_output=root_dir_output,
        run_id=run_id,
    )
