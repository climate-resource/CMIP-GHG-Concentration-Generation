"""
Test the workflow results

This is a regression test so the entire workflow is run
"""

import pandas as pd


def test_basic_workflow(basic_workflow_output_info, dataframe_regression):
    """
    Test the basic workflow

    This workflow's runtime should be kept to a minimum so it can sensibly be
    used as a test. Less than 30 seconds to run the test is what we should be
    aiming for.
    """
    for data_file in [
        basic_workflow_output_info["root_output_dir"]
        / basic_workflow_output_info["run_id"]
        / "data"
        / "processed"
        / "910_draw-table.csv"
    ]:
        data = pd.read_csv(data_file)
        dataframe_regression.check(data)
