"""
Config for retrieving, extracting and processing EPICA data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource


@frozen
class RetrieveProcessEPICAConfig:
    """
    Configuration class for retrieving and processing EPICA data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    raw_dir: Path
    """
    Directory in which the raw data is saved
    """

    download_url: URLSource
    """URL for downloading the data"""

    processed_data_with_loc_file: Path
    """File in which to save the processed data, including location information"""
