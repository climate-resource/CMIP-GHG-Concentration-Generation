"""
Config for retrieving, extracting and processing Ghosh et al. (2023) data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class RetrieveExtractGhoshEtal2023Data:
    """
    Config for retrieving, extracting and processing Ghosh et al. (2023) data

    Original paper: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022JD038281
    Downloaded from: https://www.usap-dc.org/view/dataset/601693

    Included here as there is a captcha on the download link
    which we're not going to automate around.
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    raw_data_file: Path
    """
    Path to the raw data file
    """

    expected_hash: str
    """Expected hash of the file we're loading"""

    processed_data_file: Path
    """File in which to save the processed data"""
