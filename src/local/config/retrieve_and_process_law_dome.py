"""
Config for retrieving, extracting and processing Law Dome data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen


@frozen
class RetrieveProcessLawDomeConfig:
    """
    Configuration class for retrieving and processing Law Dome data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    doi: str
    """DOI of the dataset"""

    raw_dir: Path
    """
    Directory in which the raw data is saved

    (Noting that we can't automatically download the CSIRO data)
    """

    files_md5_sum: dict[Path, str]
    """MD5 hashes for the files in the dataset"""

    processed_data_with_loc_file: Path
    """File in which to save the processed data, including location information"""
