"""
Config for retrieving, extracting and processing Scripps data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource


@frozen
class RetrieveProcessScrippsConfig:
    """
    Configuration class for retrieving and processing Scripps data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    merged_ice_core_data: URLSource
    """URLs from which to download the merged ice core data"""

    station_data: list[URLSource]
    """URLs from which to download the station data"""

    raw_dir: Path
    """
    Directory in which the raw data is saved

    (Noting that we can't automatically download the CSIRO data)
    """

    download_complete_file: Path
    """
    Path in which to write the time at which the download was completed

    This is mainly used to help with setting the dependencies between notebooks
    correctly.
    """

    processed_data_with_loc_file: Path
    """File in which to save the processed data, including location information"""
