"""
Config for retrieving and extracting ALE data
"""

from __future__ import annotations

from pathlib import Path

from attrs import frozen
from pydoit_nb.config_tools import URLSource

from local.dependencies import SourceInfo


@frozen
class RetrieveExtractALEDataConfig:
    """
    Configuration class for retrieving and extracting ALE data
    """

    step_config_id: str
    """
    ID for this configuration of the step

    Must be unique among all configurations for this step
    """

    download_urls: list[URLSource]
    """URLs from which to download the data"""

    raw_dir: Path
    """Directory in which to save the raw data"""

    download_complete_file: Path
    """
    Path in which to write the time at which the download was completed

    This is mainly used to help with setting the dependencies between notebooks
    correctly.
    """

    processed_monthly_data_with_loc_file: Path
    """
    Path in which to write the processed monthly data including location information
    """

    source_info: SourceInfo
    """Source information"""
