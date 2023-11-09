"""
Test the workflow results

This is a regression test so the entire workflow is run
"""
import os
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT_DIR = Path(__file__).parent.parent.parent


def test_basic_workflow(dataframe_regression):
    """
    Test the basic workflow

    This workflow's runtime should be kept to a minimum so it can sensibly be
    used as a test. Less than 30 seconds to run the test is what we should be
    aiming for.
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

    for data_file in [
        REPO_ROOT_DIR
        / root_output_dir
        / run_id
        / "data"
        / "processed"
        / "910_draw-table.csv"
    ]:
        data = pd.read_csv(data_file)
        dataframe_regression.check(data)
