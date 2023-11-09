"""
Configuration file for pytest
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT_DIR = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def basic_workflow_output_info() -> dict[str, str | Path]:
    """
    Run the basic workflow and get the output info
    """
    config_file = "dev-config.yaml"
    root_output_dir = "output-bundles-tests"
    run_id = "test-basic-workflow"

    subprocess.run(
        (  # noqa: S603 # inputs come from us
            "doit",
            "run",
            "--verbosity=2",
        ),
        cwd=REPO_ROOT_DIR,
        env={
            "DOIT_CONFIGURATION_FILE": config_file,
            "DOIT_ROOT_DIR_OUTPUT": root_output_dir,
            "DOIT_RUN_ID": run_id,
            **os.environ,
        },
    )

    return {
        "root_output_dir": REPO_ROOT_DIR / root_output_dir,
        "run_id": run_id,
    }
